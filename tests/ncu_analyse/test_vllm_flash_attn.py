import platform; 
system= platform.system().lower()

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
    # from vllm_flash_attn import flash_attn_func 
except ImportError as e:
    raise ImportError("VLLM Flash Attention not found") from e

if system == "linux":
    try:
        from tile_flash_attn import flashattn, do_bench 
    except ImportError as e:
        tilelang_source = """
        https://github.com/tile-ai/tilelang/
        https://github.com/tile-ai/tilelang/releases/download/v0.1.8/tilelang-0.1.8-cp38-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
        """
        print(f"Tile Flash Attention not found, {e}")
        print(f"Tilelang source: \n{tilelang_source}")
        from triton.testing import do_bench
else :
    from triton.testing import do_bench

import torch, math, os
import torch.nn.functional as F
# from torch.nn.attention import SDPBackend, sdpa_kernel
import argparse
import matplotlib.pyplot as plt


def ref_program(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, is_causal: bool = False):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, 
        attn_mask=mask, dropout_p=0.0, is_causal=is_causal)

def ref_program_fa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal=False):
    return flash_attn_func(
        query.permute(0,2,1,3), key.permute(0,2,1,3), value.permute(0,2,1,3),
        causal=causal
        ).permute(0,2,1,3)

def ref_program_kvcache(
    q: torch.Tensor,          # (batch, 1, nheads, head_dim)  or (batch, seqlen_q, nheads, head_dim)
    k: torch.Tensor,          # (batch, seqlen_k, nheads_k, head_dim) — full keys to fill cache
    v: torch.Tensor,          # (batch, seqlen_k, nheads_k, head_dim) — full values to fill cache
    cache_seqlens: torch.Tensor,  # (batch,) int32, actual lengths
    v_cache_is_transposed: bool = False,
    causal: bool = False,
):
    """Reference using flash_attn_func on the full K/V, compared against flash_attn_with_kvcache."""
    out_ref = flash_attn_func(
        q.permute(0,2,1,3), k.permute(0,2,1,3), v.permute(0,2,1,3),
        causal=causal
    ).permute(0,2,1,3)

    # Allocate KV cache.
    # K cache is always (B,S,H,D). V cache is (B,H,D,S) when v_cache_is_transposed=True.
    max_seqlen_k = k.size(1)
    k_cache = torch.zeros(q.size(0), max_seqlen_k, k.size(2), k.size(3),
                          dtype=k.dtype, device=k.device)
    k_cache[:, :k.size(1)] = k
    if v_cache_is_transposed:
        v_cache = torch.zeros(q.size(0), k.size(2), k.size(3), max_seqlen_k,
                              dtype=k.dtype, device=k.device)
        v_cache[:, :, :, :v.size(1)] = v.permute(0, 1, 3, 2)  # (B,H,S,D) → (B,H,D,S)
    else:
        v_cache = torch.zeros_like(k_cache)
        v_cache[:, :v.size(1)] = v

    out_kvcache, _ = flash_attn_with_kvcache(
        q.permute(0,2,1,3), k_cache, v_cache,
        cache_seqlens=cache_seqlens, causal=causal,
        v_cache_is_transposed=v_cache_is_transposed,
    )
    out_kvcache = out_kvcache.permute(0,2,1,3)

    diff = (out_ref - out_kvcache).abs()
    print(f"  KVCache diff: max={diff.max().item():.6f} mean={diff.mean().item():.6f}")
    return out_kvcache

def manual_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = torch.nn.functional.softmax(att, dim=-1)
    y = torch.matmul(att, v)
    return y

# def benchmark(f, *args, **kwargs):
#     return (do_bench(lambda: f(*args, **kwargs), return_mode="mean") * 1e3)  # return in us

# region torch profile test
def profile_function(name, func, *args, **kwargs):
    for i in range(5):
        _ = func(*args, **kwargs); torch.cuda.synchronize()
    print(f'=== profiling {name} ===')

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True, profile_memory=True, # use_cuda=True,
    ) as prof:
        for _ in range(5):
            func(*args, **kwargs); torch.cuda.synchronize()

        with torch.profiler.record_function(f"{name}_event"):
            result = func(*args, **kwargs); torch.cuda.synchronize()
    torch.cuda.empty_cache()
    filtered_output = '\n'.join(
        line for line in prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=200).split('\n') if '---' not in line)
    print(filtered_output)
    print(result.shape, result[0][0][0][0:8], "\n")
    return result

def torch_profile(
    args, q_shape, kv_shape,
    block_M=128, block_N=64, num_stages=1, threads=256, is_causal=False, attn_mask=False):

    causal, dtype, device = is_causal, torch.float16, "cuda"
    print(f"torch: {torch.__version__}, cudnn: {torch.backends.cudnn.version()}, name: {torch.cuda.get_device_properties(0).name}")

    query = torch.randn(q_shape, dtype=dtype, device=device)
    key = torch.randn(kv_shape, dtype=dtype, device=device)
    value = torch.randn(kv_shape, dtype=dtype, device=device)
    sm_scale = (1.0 / (q_shape[3] ** 0.5))

    print("Q BATCH:", query.shape[0], "Q N_HEADS:", query.shape[1], "Q SEQ_LEN:", query.shape[2], "Q HEAD_DIM:", query.shape[3], "dtype:", dtype, "device:", device)
    print("kV BATCH:", key.shape[0], "kV N_HEADS:", key.shape[1], "kV SEQ_LEN:", key.shape[2], "kV HEAD_DIM:", key.shape[3])

    # manual_result = profile_function("manual attention", manual_attn, query, key, value)

    tile_flash_result = None; sdpa_result = None; vllm_flash_result = None
    if args.fa:
        vllm_flash_result = profile_function("vllm flash attention", ref_program_fa, query, key, value, causal=causal)
    if args.tile:
        kernel = flashattn(
        query.shape,
        key.shape,
        is_causal=is_causal,
        attn_mask=attn_mask,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads)
        tile_flash_result = profile_function("tile flash attention", kernel, query, key, value)

    if args.sdpa:
        torch.backends.cuda.enable_math_sdp(False)  # use Torch
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # use xformers
        torch.backends.cuda.enable_flash_sdp(False)   # use flash attn backend
        # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        sdpa_result = profile_function("sdpa", F.scaled_dot_product_attention, query, key, value, is_causal=causal, scale=sm_scale)

    if vllm_flash_result is not None and sdpa_result is not None:
        diff = torch.abs(vllm_flash_result - sdpa_result)
        print(f"VLLM Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n", flush=True)
        if not torch.allclose(vllm_flash_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("VLLM Flash Attention and SDPA results do not match.")

    if tile_flash_result is not None and sdpa_result is not None:
        diff = torch.abs(tile_flash_result - sdpa_result)
        print(f"Tile Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n", flush=True)
        if not torch.allclose(tile_flash_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("Tile Flash Attention and SDPA results do not match.")

# ==========================================================================================

# region flops test
plot_data = {
    'q_shapes': [],
    'kv_shapes': [],
    'fa_tflops': [],
    'fa_latency': [],
    'tile_tflops': [],
    'tile_latency': [],
    'sdpa_tflops': [],
    'sdpa_latency': [],
}

def flops(
    args, num, q_shape, kv_shape,
    block_M=128, block_N=64, num_stages=1, threads=256, 
    is_causal: bool = False, attn_mask: bool = False, 
):

    flops_per_matmul = 2.0 * q_shape[0] * q_shape[1] * q_shape[2] * kv_shape[2] * q_shape[3]
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    dtype, device = torch.float16, "cuda"
    query = torch.randn(q_shape, dtype=dtype, device=device)
    key = torch.randn(kv_shape, dtype=dtype, device=device)
    value = torch.randn(kv_shape, dtype=dtype, device=device)
    
    if attn_mask:
        mask = torch.zeros(q_shape[0], q_shape[1], q_shape[2], kv_shape[2], dtype=dtype, device=device)
        mask[:, :, :, kv_shape[2]//2:] = -torch.inf # 屏蔽后半部分 keys
    else:
        mask = None
    
    print("Q BATCH:", q_shape[0], "Q N_HEADS:", q_shape[1], "Q SEQ_LEN:", q_shape[2], "Q HEAD_DIM:", q_shape[3], "dtype:", dtype, "device:", device)
    print("KV_BATCH:", kv_shape[0], "KV_N_HEADS:", kv_shape[1], "KV_SEQ_LEN:", kv_shape[2], "KV_HEAD_DIM:", kv_shape[3])

    sdpa_result = None; tile_result = None; fa_result = None
    if args.fa:
        fa_result = ref_program_fa(query, key, value)
    if args.sdpa:
        sdpa_result = ref_program(query, key, value, mask, is_causal)
    
    if args.tile:
        kernel = flashattn(
            q_shape,
            kv_shape,
            is_causal,
            attn_mask, 
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads)    
        
        if attn_mask:
            tile_result = kernel(query, key, value, mask)
        else:   
            tile_result = kernel(query, key, value)

        if args.show_tile_source:
            with open(os.path.join(os.path.dirname(__file__), f"tile_flash_attn_dim{q_shape[3]}.cu"), "w") as f:
                f.write(kernel.get_kernel_source())
                print(f"tile_flash_attn_dim{q_shape[3]}.cu saved.")
    
    if sdpa_result is not None and fa_result is not None:
        diff = torch.abs(fa_result - sdpa_result)
        print(f"VLLM Flash Attention Ref torch diff max: {diff.max()}, mean: {diff.mean()}\n")
        if not torch.allclose(fa_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("VLLM Flash Attention and Ref torch results do not match.")
    
    if tile_result is not None and sdpa_result is not None:
        diff = torch.abs(tile_result - sdpa_result)
        print(f"Tile Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n")
        if not torch.allclose(tile_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("Tile Flash Attention and Ref torch results do not match.")

    fa_latency = None; tile_latency = None; sdpa_latency = None
    fa_tflops = None; tile_tflops = None; sdpa_tflops = None
    
    if args.fa:
        fa_latency = do_bench(fn=lambda: ref_program_fa(query, key, value), rep=num, warmup=int(num/2))
        fa_tflops = total_flops / fa_latency * 1e-9
        print(f"VLLM Flash Attention: {fa_latency:.3f} ms")
        print(f"VLLM Flash Attention: {fa_tflops:.3f} TFlops")

    if args.tile:
        if attn_mask:
            tile_latency = do_bench(fn=lambda: kernel(query, key, value, mask), rep=num, warmup=int(num/2))
        else:   
            tile_latency = do_bench(fn=lambda: kernel(query, key, value), rep=num, warmup=int(num/2))
        tile_tflops = total_flops / tile_latency * 1e-9
        print(f"Tile Flash Attention: {tile_latency:.3f} ms")
        print(f"Tile Flash Attention: {tile_tflops:.3f} TFlops")
    
    if args.sdpa :
        sdpa_latency = do_bench(fn=lambda: ref_program(query, key, value, mask, is_causal), rep=num, warmup=int(num/2))
        sdpa_tflops = total_flops / sdpa_latency * 1e-9
        print(f"Ref SDPA: {sdpa_latency:.3f} ms")
        print(f"Ref SDPA: {sdpa_tflops:.3f} TFlops")

    if (fa_result is not None) and (sdpa_result is not None) :
        vllm_fa_speed_up = (1 - (float(fa_latency) / float(sdpa_latency)) ) * 100
        torch.testing.assert_close(fa_result, sdpa_result, rtol=5e-4, atol=5e-4)
        print(f"fa result checks pass. fa speed up: {vllm_fa_speed_up:.2f}%\n")
    
    if (tile_result is not None) and (sdpa_result is not None) :
        tile_speed_up = (1 - (float(tile_latency) / float(sdpa_latency)) ) * 100
        torch.testing.assert_close(tile_result, sdpa_result, rtol=5e-4, atol=5e-4)
        print(f"tile result checks pass. tile speed up: {tile_speed_up:.2f}%\n")
    
    if args.plot:
        plot_data['q_shapes'].append(q_shape)
        plot_data['kv_shapes'].append(kv_shape)
        if fa_tflops is not None:
            plot_data['fa_tflops'].append(fa_tflops)
            plot_data['fa_latency'].append(fa_latency)
        if tile_tflops is not None:
            plot_data['tile_tflops'].append(tile_tflops)
            plot_data['tile_latency'].append(tile_latency)
        if sdpa_tflops is not None:
            plot_data['sdpa_tflops'].append(sdpa_tflops)
            plot_data['sdpa_latency'].append(sdpa_latency)


def kvcache_flops(
    args, num, q_shape, kv_shape,
    v_cache_is_transposed: bool = False,
    is_causal: bool = False,
):
    """Test flash_attn_with_kvcache — inference / KV-cache decode.

    Pre-fills K/V cache and runs decode (seqlen_q) attention through the
    KV-cache interface, comparing against flash_attn_func on the full K/V.
    """
    batch, heads_q, seqlen_q, head_dim = q_shape
    _, heads_kv, seqlen_k, _ = kv_shape

    flops_per_matmul = 2.0 * batch * heads_q * seqlen_q * seqlen_k * head_dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        # FA uses right-aligned causal: query i attends to keys [0, seqlen_k - seqlen_q + i].
        # attended = seqlen_q * seqlen_k - seqlen_q*(seqlen_q-1)/2.
        # For decode (seqlen_q=1) this is full attention (ratio=1); for prefill
        # (seqlen_q==seqlen_k) it reduces to the standard 0.5 triangle.
        total_flops *= 1.0 - (seqlen_q - 1) / (2.0 * seqlen_k)

    dtype, device = torch.float16, "cuda"
    query = torch.randn(q_shape, dtype=dtype, device=device)
    key   = torch.randn(kv_shape, dtype=dtype, device=device)
    value = torch.randn(kv_shape, dtype=dtype, device=device)

    print(f"KVCache: Q {batch}x{heads_q}x{seqlen_q}x{head_dim}, "
          f"KV {batch}x{heads_kv}x{seqlen_k}x{head_dim}, "
          f"transposed={v_cache_is_transposed}")

    # Reference using flash_attn_func on full K/V
    ref_out = ref_program_fa(query, key, value, causal=is_causal)

    # Allocate KV cache.
    # K cache is always (B,S,H,D). V cache layout depends on v_cache_is_transposed:
    #   False → (B,S,H,D) standard layout
    #   True  → (B,H,D,S) pre-transposed layout (caller responsibility)
    # key/value are (B,H,S,D), so transpose dims 1,2 for K, and permute(0,1,3,2) for transposed V.
    k_cache = torch.zeros(batch, seqlen_k, heads_kv, head_dim, dtype=dtype, device=device)
    k_cache.copy_(key.transpose(1, 2))
    if v_cache_is_transposed:
        v_cache = torch.zeros(batch, heads_kv, head_dim, seqlen_k, dtype=dtype, device=device)
        v_cache.copy_(value.permute(0, 1, 3, 2))  # (B,H,S,D) → (B,H,D,S)
    else:
        v_cache = torch.zeros_like(k_cache)
        v_cache.copy_(value.transpose(1, 2))

    # For decode, cache_seqlens = full seqlen_k (no partial usage for simplicity)
    cache_seqlens = torch.full((batch,), seqlen_k, dtype=torch.int32, device=device)

    # Warmup
    for _ in range(5):
        _ = flash_attn_with_kvcache(
            query.permute(0,2,1,3), k_cache, v_cache,
            cache_seqlens=cache_seqlens, causal=is_causal,
            v_cache_is_transposed=v_cache_is_transposed,
        )
        torch.cuda.synchronize()

    # Benchmark
    fa_latency = do_bench(
        fn=lambda: flash_attn_with_kvcache(
            query.permute(0,2,1,3), k_cache, v_cache,
            cache_seqlens=cache_seqlens, causal=is_causal,
            v_cache_is_transposed=v_cache_is_transposed,
        ),
        rep=num, warmup=int(num/2)
    )

    # SDPA causal flag: FA right-aligns the causal mask when seqlen_q < seqlen_k
    # (decode), so query at the end attends to all cached keys. PyTorch SDPA
    # is_causal=True left-aligns (query at position 0 → attends 1 key), which
    # computes a different result. Only use is_causal=True for SDPA when the
    # masks actually coincide (seqlen_q >= seqlen_k).
    sdpa_is_causal = is_causal and seqlen_q >= seqlen_k

    # torch profiler for kvcache
    if args.profile:
        profile_function(
            "kvcache flash attention",
            flash_attn_with_kvcache,
            query.permute(0,2,1,3), k_cache, v_cache,
            cache_seqlens=cache_seqlens, causal=is_causal,
            v_cache_is_transposed=v_cache_is_transposed,
        )
        if args.sdpa:
            profile_function(
                "kvcache ref sdpa",
                ref_program, query, key, value, None, sdpa_is_causal,
            )

    # Correctness: compare kvcache output vs flash_attn_func reference
    out_kvcache = flash_attn_with_kvcache(
        query.permute(0,2,1,3), k_cache, v_cache,
        cache_seqlens=cache_seqlens, causal=is_causal,
        v_cache_is_transposed=v_cache_is_transposed,
    )    
    fa_tflops = total_flops / fa_latency * 1e-9
    print(f"  KVCache FA: {fa_latency:.3f} ms, {fa_tflops:.2f} TFlops")

    out_kvcache = out_kvcache.permute(0,2,1,3)
    diff = (ref_out - out_kvcache).abs()
    print(f"  KVCache vs ref: max_diff={diff.max().item():.6f} mean_diff={diff.mean().item():.6f}")
    if not torch.allclose(ref_out, out_kvcache, atol=5e-4, rtol=5e-4):
        print("  *** WARNING: KVCache output does NOT match reference! ***")
    else:
        print("  KVCache correctness PASSED")

    # SDPA reference for comparison
    if args.sdpa:
        sdpa_result = ref_program(query, key, value, None, sdpa_is_causal)
        sdpa_latency = do_bench(
            fn=lambda: ref_program(query, key, value, None, sdpa_is_causal),
            rep=num, warmup=int(num/2)
        )
        sdpa_tflops = total_flops / sdpa_latency * 1e-9
        print(f"  Ref SDPA: {sdpa_latency:.3f} ms, {sdpa_tflops:.2f} TFlops")

        # Correctness: kvcache vs SDPA
        sdpa_diff = (ref_out - sdpa_result).abs()
        print(f"  KVCache vs SDPA: max_diff={sdpa_diff.max().item():.6f} mean_diff={sdpa_diff.mean().item():.6f}")
        if not torch.allclose(ref_out, sdpa_result, atol=5e-4, rtol=5e-4):
            print("  *** WARNING: KVCache and SDPA results do NOT match! ***")

        kvcache_speedup = (1 - fa_latency / sdpa_latency) * 100
        print(f"  KVCache vs SDPA speedup: {kvcache_speedup:.1f}%\n")


# region run test
if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    parser = argparse.ArgumentParser(description="Test VLLM Flash Attention")
    parser.add_argument("--profile", action="store_true", help="Profile the kernel")
    parser.add_argument("--flops", action="store_true", help="Test the flops")
    parser.add_argument("--flops_num", type=int, default=500, help="Number of flops test")
    parser.add_argument("--show_tile_source", action="store_true", help="Show the tilelang cuda source code")

    parser.add_argument("--fa", action="store_true", help="Use the flash attention")
    parser.add_argument("--sdpa", action="store_true", help="Use the SDPA results")
    parser.add_argument("--tile", action="store_true", help="Use the tile flash attention")

    # KV cache (inference / decode) test flags
    parser.add_argument("--kvcache", action="store_true",
                        help="Test flash_attn_with_kvcache instead of flash_attn_func. "
                             "KV cache is pre-filled with K/V; q is decode (seqlen=1 unless overridden).")
    parser.add_argument("--kvcache_transposed", action="store_true",
                        help="Enable V pre-transpose optimization in the kvcache test "
                             "(requires seqlen_k %% 128 == 0, head_dim %% 32 == 0).")

    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--n_heads", type=int, default=40, help="Number of heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    
    parser.add_argument("--plot", action="store_true", help="Plot q_shape, kv_shape, tflops and time charts when running flops test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--multi", action="store_true", help="Only test once")
    
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    q_shape_list = []
    kv_shape_list = []

    # ============================================================
    # KV cache (inference / decode) path
    # ============================================================
    if args.kvcache:
        if not args.multi:
            # Single-shape: use CLI batch/heads/seq_len/head_dim.
            # Default to decode: q seqlen=1, kv cache = full seq_len.
            q_shape = (args.batch, args.n_heads, 1, args.head_dim)  # decode: 1 token
            # Ensure kv cache seqlen is a multiple of 128 for v_cache_is_transposed
            seqlen_k = (args.seq_len + 127) // 128 * 128
            kv_shape = (args.batch, args.n_heads, seqlen_k, args.head_dim)
            q_shape_list.append(q_shape)
            kv_shape_list.append(kv_shape)
        else:
            # Multi-shape: predefined decode benchmarks.
            # Each (q_shape, kv_shape) — q is decode (1 token) or prefill.
            _h = args.head_dim
            kvcache_cases = [
                # (q_shape, kv_shape) — decode: q seqlen=1
                ((2, 40, 1, _h),    (2, 40, 65536, _h)),
                ((2, 40, 1, _h),    (2, 40, 131072, _h)),
                # ((8, 10, 1, _h),    (8, 10, 4096, _h)),
                # ((8, 10, 1, _h),    (8, 10, 8192, _h)),
                # Prefill-like: q seqlen > 1
                # ((2, 16, 1024, _h),  (2, 16, 1024, _h)),
                # ((2, 16, 4096, _h),  (2, 16, 4096, _h)),
            ]
            for qs, kvs in kvcache_cases:
                q_shape_list.append(qs)
                kv_shape_list.append(kvs)

        for q_shape, kv_shape in zip(q_shape_list, kv_shape_list):
            kvcache_flops(
                args, is_causal=True, num=args.flops_num,
                q_shape=q_shape, kv_shape=kv_shape,
                v_cache_is_transposed=args.kvcache_transposed,
            )
    else:
        # ============================================================
        # Standard flash_attn_func / flops / profile path
        # ============================================================
        if not args.multi:
            q_shape = (args.batch, args.n_heads, args.seq_len, args.head_dim) 
            kv_shape = (args.batch, args.n_heads, args.seq_len, args.head_dim) 
            q_shape_list.append(q_shape)
            kv_shape_list.append(kv_shape)
        else:
            q_shape_list.append((1, 40, 1, args.head_dim))
            kv_shape_list.append((1, 40, 4096, args.head_dim))

            q_shape_list.append((1, 40, 1, args.head_dim))
            kv_shape_list.append((1, 40, 32768, args.head_dim))

            q_shape_list.append((1, 40, 1, args.head_dim))
            kv_shape_list.append((1, 40, 65536, args.head_dim))

            q_shape_list.append((1, 40, 4096, args.head_dim))
            kv_shape_list.append((1, 40, 4096, args.head_dim))

            # q_shape_list.append((1, 40, 16384, args.head_dim))
            # kv_shape_list.append((1, 40, 16384, args.head_dim))

            # q_shape_list.append((2, 20, 1024, args.head_dim))
            # kv_shape_list.append((2, 20, 77, args.head_dim))

            # q_shape_list.append((2, 10, 4096, args.head_dim))
            # kv_shape_list.append((2, 10, 77, args.head_dim))

            # q_shape_list.append((2, 10, 6144, args.head_dim))
            # kv_shape_list.append((2, 10, 77, args.head_dim))

            # q_shape_list.append((2, 16, 3952,  args.head_dim))
            # kv_shape_list.append((2, 16, 512,  args.head_dim))

            # q_shape_list.append((2, 16, 3952,  args.head_dim))
            # kv_shape_list.append((2, 16, 3952,  args.head_dim))

            # q_shape_list.append((8, 10, 512, args.head_dim))
            # kv_shape_list.append((8, 10, 512, args.head_dim))

            # q_shape_list.append((8, 10, 1024, args.head_dim))
            # kv_shape_list.append((8, 10, 1024, args.head_dim))

            # q_shape_list.append((8, 10, 2048, args.head_dim))
            # kv_shape_list.append((8, 10, 2048, args.head_dim))

            q_shape_list.append((2, 40, 4096, args.head_dim))
            kv_shape_list.append((2, 40, 4096, args.head_dim))

            q_shape_list.append((2, 40, 16384, args.head_dim))
            kv_shape_list.append((2, 40, 16384, args.head_dim))


        for q_shape, kv_shape in zip(q_shape_list, kv_shape_list):
            if args.profile:
                torch_profile(
                    args,
                    q_shape, kv_shape,
                    block_M=128, block_N=64, num_stages=1, threads=256, is_causal=False, attn_mask=False
                    )

            if args.flops:
                flops(
                    args, is_causal=False, attn_mask=False, num = args.flops_num,
                    q_shape=q_shape, kv_shape=kv_shape,
                    block_M=128, block_N=64, num_stages=1, threads=256, 
                )

        if args.flops and args.plot and plot_data['q_shapes']:
            print("\n=== Generating plots ===")
            x_labels_q = [str(tuple(q)) for q in plot_data['q_shapes']]
            x_labels_kv = [str(tuple(kv)) for kv in plot_data['kv_shapes']]
            x_labels = [f"{q}\n{kv}" for q, kv in zip(x_labels_q, x_labels_kv)]
            x = range(len(x_labels))

            # 根据数据点数量动态调整柱宽和图表大小
            num_points = len(x_labels)
            width = 0.8 / (len([k for k in ['fa_tflops', 'tile_tflops', 'sdpa_tflops'] if plot_data[k]]) + 1)
            figsize_width = min(18, max(12, num_points * 2.5))

            colors = {
                'fa': '#1f77b4',      # blue
                'tile': '#ff7f0e',    # orange
                'sdpa': '#2ca02c',    # green
            }

            # 动态调整字体大小
            label_fontsize = max(6, min(9, 12 - num_points))
            value_fontsize = max(5, min(8, 10 - num_points))

            fig, axes = plt.subplots(1, 2, figsize=(figsize_width, 6))

            # TFLOPs chart
            ax1 = axes[0]
            offset1 = 0
            if plot_data['fa_tflops']:
                rects = ax1.bar([i + offset1 for i in x], plot_data['fa_tflops'], width, 
                               label='FlashAttention', color=colors['fa'], edgecolor='black', linewidth=0.5)
                for rect in rects:
                    height = rect.get_height()
                    ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                               ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
                offset1 += width
            if plot_data['tile_tflops']:
                rects = ax1.bar([i + offset1 for i in x], plot_data['tile_tflops'], width,
                               label='TileLang', color=colors['tile'], edgecolor='black', linewidth=0.5)
                for rect in rects:
                    height = rect.get_height()
                    ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                               ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
                offset1 += width
            if plot_data['sdpa_tflops']:
                rects = ax1.bar([i + offset1 for i in x], plot_data['sdpa_tflops'], width,
                               label='SDPA', color=colors['sdpa'], edgecolor='black', linewidth=0.5)
                for rect in rects:
                    height = rect.get_height()
                    ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                               ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')

            ax1.set_ylabel('Speed (TFLOP/s)', fontsize=12)
            ax1.set_xlabel('Sequence length', fontsize=12)
            ax1.set_title(f'v100 Attention forward TFLOPs hdim {args.head_dim}', fontsize=13)
            ax1.set_xticks([i + width/2 for i in x])
            ax1.set_xticklabels(x_labels, fontsize=label_fontsize, rotation=0)
            ax1.legend(loc='upper left', fontsize=10)
            ax1.set_axisbelow(True)
            ax1.grid(True, axis='y', linestyle='--', alpha=0.3)

            # Latency chart
            ax2 = axes[1]
            offset2 = 0
            if plot_data['fa_latency']:
                rects = ax2.bar([i + offset2 for i in x], plot_data['fa_latency'], width,
                               label='FlashAttention', color=colors['fa'], edgecolor='black', linewidth=0.5)
                for rect in rects:
                    height = rect.get_height()
                    ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                               ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
                offset2 += width
            if plot_data['tile_latency']:
                rects = ax2.bar([i + offset2 for i in x], plot_data['tile_latency'], width,
                               label='TileLang', color=colors['tile'], edgecolor='black', linewidth=0.5)
                for rect in rects:
                    height = rect.get_height()
                    ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                               ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
                offset2 += width
            if plot_data['sdpa_latency']:
                rects = ax2.bar([i + offset2 for i in x], plot_data['sdpa_latency'], width,
                               label='SDPA', color=colors['sdpa'], edgecolor='black', linewidth=0.5)
                for rect in rects:
                    height = rect.get_height()
                    ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                               ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')

            ax2.set_ylabel('Latency (ms)', fontsize=12)
            ax2.set_xlabel('Sequence length', fontsize=12)
            ax2.set_title(f'v100 Attention latency hdim {args.head_dim}', fontsize=13)
            ax2.set_xticks([i + width/2 for i in x])
            ax2.set_xticklabels(x_labels, fontsize=label_fontsize, rotation=0)
            ax2.legend(loc='upper left', fontsize=10)
            ax2.set_axisbelow(True)
            ax2.grid(True, axis='y', linestyle='--', alpha=0.3)

            save_plot = f'flops_{str(q_shape_list[-1]).replace(" ", "")}_{str(kv_shape_list[-1]).replace(" ", "")}.png'
            plt.tight_layout(); 
            plt.savefig(save_plot, dpi=150, bbox_inches='tight')
            print(f"Plot saved as {save_plot}")
            plt.show()


Pytorch_AttentionKernel_Generator = """
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.q),
                str(self.k),
                str(self.max_k),
                "true" if self.supports_dropout else "false",
                "true" if self.supports_bias else "false",
            ]
        )
        return f"AttentionKernel<{template_args}>"
"""

# Standard flash_attn_func:
# python tests/ncu_analyse/test_vllm_flash_attn.py --flops --fa --sdpa --head_dim 128 
# python tests/ncu_analyse/test_vllm_flash_attn.py --flops --profile --fa --sdpa --head_dim 64

# KV cache / decode (flash_attn_with_kvcache, triggers SplitKV path):
# python tests/ncu_analyse/test_vllm_flash_attn.py --kvcache --fa --head_dim 128 --batch 2 --n_heads 40 --seq_len 4096
# python tests/ncu_analyse/test_vllm_flash_attn.py --kvcache --multi --fa --sdpa --head_dim 128
#
# With V pre-transpose (requires --head_dim 64 or 128, seq_len multiple of 128):
# python tests/ncu_analyse/test_vllm_flash_attn.py --kvcache --kvcache_transposed --multi --fa --sdpa --head_dim 128
