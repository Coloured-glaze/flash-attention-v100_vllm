import torch
import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

@tilelang.jit(
    out_idx=[-1], pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    })
def flashattn(
              q_shape,
              kv_shape,
              is_causal,
              attn_mask,
              block_M=128,
              block_N=64,
              num_stages=1,
              threads=256):
    batch = q_shape[0]
    heads = q_shape[1]
    seq_q = q_shape[2]
    q_dim = q_shape[3]
    seq_kv = kv_shape[2]
    kv_dim = kv_shape[3]
    
    # 预计算常数
    scale_val = (1.0 / q_dim)**0.5
    scale_log2e = scale_val * 1.4426950408889634

    if attn_mask:
        mask_shape = [batch, heads, seq_q, seq_kv]

    dtype = "float16"
    accum_dtype = "float32" 

    past_len = seq_kv - seq_q

    # --- Macros ---

    @T.macro
    def MMA0(
        K: T.Tensor(kv_shape, dtype),
        Q_shared: T.SharedBuffer([block_M, q_dim], dtype),
        K_shared: T.SharedBuffer([block_N, kv_dim], dtype),
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
        bx: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(K[bz, by, k * block_N:(k + 1) * block_N, :], K_shared)
        if is_causal:
            for i, j in T.Parallel(block_M, block_N):
                q_idx = bx * block_M + i + past_len
                k_idx = k * block_N + j
                # Causal Mask 逻辑：
                # 如果 q_idx >= k_idx，初始化为 0；否则 -inf。
                # 注意：这里隐式处理了部分边界，但建议后续统一加上 seq_kv 边界检查更安全
                acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
        else:
            # We shall fill -inf for OOB positions
            for i, j in T.Parallel(block_M, block_N):
                acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
        T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def MMA1(
        V: T.Tensor(kv_shape, dtype),
        V_shared: T.SharedBuffer([block_N, kv_dim], dtype),
        acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
        acc_o: T.FragmentBuffer([block_M, kv_dim], accum_dtype),
        k: T.int32,
        by: T.int32,
        bz: T.int32,
    ):
        T.copy(V[bz, by, k * block_N:(k + 1) * block_N, :], V_shared)
        T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

    # 修复1: 增强的 ApplyMask，同时处理 User Mask 和 KV 长度边界
    if attn_mask:
        @T.macro
        def ApplyMask(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            mask: T.Tensor(mask_shape, dtype),
            mask_shared: T.SharedBuffer([block_M, block_N], dtype),
            k: T.int32,
            bx: T.int32,
            by: T.int32,
            bz: T.int32,
        ):
            T.copy(mask[bz, by, bx * block_M:(bx + 1) * block_M, k * block_N:(k + 1) * block_N], mask_shared)
            for i, j in T.Parallel(block_M, block_N):
                k_idx = k * block_N + j
                # 关键修复：如果 k_idx 越界，强制设为 -inf。
                # 否则，加上 Mask 的值 (0 或 -inf)。
                # 这避免了 T.copy 在越界时填充 0 导致的错误 Attention。
                acc_s[i, j] = T.if_then_else(
                    k_idx < seq_kv, 
                    acc_s[i, j] + mask_shared[i, j], 
                    -T.infinity(acc_s.dtype)
                )

    # 修复2: 为无 Mask 模式增加边界检查
    @T.macro
    def ApplyBoundary(
        acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
        k: T.int32,
    ):
        for i, j in T.Parallel(block_M, block_N):
            k_idx = k * block_N + j
            if k_idx >= seq_kv:
                acc_s[i, j] = -T.infinity(acc_s.dtype)

    @T.macro
    def Softmax(
            acc_s: T.FragmentBuffer([block_M, block_N], accum_dtype),
            acc_s_cast: T.FragmentBuffer([block_M, block_N], dtype),
            scores_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_max_prev: T.FragmentBuffer([block_M], accum_dtype),
            block_max: T.FragmentBuffer([block_M], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
            scores_sum: T.FragmentBuffer([block_M], accum_dtype),
            logsum: T.FragmentBuffer([block_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        
        T.fill(block_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, block_max, dim=1, clear=False)
        
        for i in T.Parallel(block_M):
            scores_max[i] = T.max(scores_max_prev[i], block_max[i])

        for i in T.Parallel(block_M):
            prev = scores_max_prev[i]
            curr = scores_max[i]
            # 数值稳定性修复：如果 prev 是 -inf (初始状态或之前全被Mask)，Scale 设为 1.0 (避免 NaN)
            # 实际上此时 LogSum 为 0，Scale 不影响结果，但计算必须安全。
            diff = T.if_then_else(prev == -T.infinity(accum_dtype), 0.0, prev - curr)
            scores_scale[i] = T.exp2(diff * scale_log2e)

        for i, j in T.Parallel(block_M, block_N):
            val = acc_s[i, j]
            m = scores_max[i]
            # 处理 -inf 输入
            acc_s[i, j] = T.if_then_else(val == -T.infinity(accum_dtype), 0.0, T.exp2((val - m) * scale_log2e))

        T.fill(scores_sum, 0)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
            
        T.copy(acc_s, acc_s_cast)

    @T.macro
    def Rescale(
            acc_o: T.FragmentBuffer([block_M, kv_dim], accum_dtype),
            scores_scale: T.FragmentBuffer([block_M], accum_dtype),
    ):
        for i, j in T.Parallel(block_M, kv_dim):
            acc_o[i, j] *= scores_scale[i]

    if attn_mask:
        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                V: T.Tensor(kv_shape, dtype),
                Mask: T.Tensor(mask_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, q_dim], dtype)
                K_shared = T.alloc_shared([block_N, kv_dim], dtype)
                V_shared = T.alloc_shared([block_N, kv_dim], dtype)
                O_shared = T.alloc_shared([block_M, kv_dim], dtype)
                Mask_shared = T.alloc_shared([block_M, block_N], dtype)
                
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, kv_dim], accum_dtype)
                
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                block_max = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(
                        T.ceildiv(seq_kv, block_N), T.ceildiv(
                            (bx + 1) * block_M +
                            past_len, block_N)) if is_causal else T.ceildiv(seq_kv, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]]):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    ApplyMask(acc_s, Mask, Mask_shared, k, bx, by, bz) # 内部包含边界检查
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max, scores_scale, scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                
                for i, j in T.Parallel(block_M, kv_dim):
                    # 归一化，避免除以0
                    inv_logsum = T.if_then_else(logsum[i] > 1e-6, 1.0 / logsum[i], 0.0)
                    acc_o[i, j] *= inv_logsum
                
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])
                
    else:
        @T.prim_func
        def main(
                Q: T.Tensor(q_shape, dtype),
                K: T.Tensor(kv_shape, dtype),
                V: T.Tensor(kv_shape, dtype),
                Output: T.Tensor(q_shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, q_dim], dtype)
                K_shared = T.alloc_shared([block_N, kv_dim], dtype)
                V_shared = T.alloc_shared([block_N, kv_dim], dtype)
                O_shared = T.alloc_shared([block_M, kv_dim], dtype)
                
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_shared([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, kv_dim], accum_dtype)
                
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                block_max = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, by, bx * block_M:(bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = (
                    T.min(
                        T.ceildiv(seq_kv, block_N), T.ceildiv(
                            (bx + 1) * block_M +
                            past_len, block_N)) if is_causal else T.ceildiv(seq_kv, block_N))

                for k in T.Pipelined(
                        loop_range,
                        num_stages=num_stages,
                        order=[-1, 0, 3, 1, -1, 2],
                        stage=[-1, 0, 0, 1, -1, 1],
                        group=[[0], [1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11], [12], [13], [14]]):
                    MMA0(K, Q_shared, K_shared, acc_s, k, bx, by, bz)
                    if not is_causal:
                        ApplyBoundary(acc_s, k) # 显式应用边界 Mask
                    Softmax(acc_s, acc_s_cast, scores_max, scores_max_prev, block_max, scores_scale, scores_sum, logsum)
                    Rescale(acc_o, scores_scale)
                    MMA1(V, V_shared, acc_s_cast, acc_o, k, by, bz)
                
                for i, j in T.Parallel(block_M, kv_dim):
                    inv_logsum = T.if_then_else(logsum[i] > 1e-6, 1.0 / logsum[i], 0.0)
                    acc_o[i, j] *= inv_logsum
                
                T.copy(acc_o, O_shared)
                T.copy(O_shared, Output[bz, by, bx * block_M:(bx + 1) * block_M, :])

    return main

