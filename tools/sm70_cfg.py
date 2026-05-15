#!/usr/bin/env python3
"""
V100 SM70 Flash Attention Configuration Calculator (Optimized)
- Uses estimated TFLOPS for ranking (not weighted score)
- Generates dispatch table for different seqlen ranges
- Supports head dims 64, 96, 128, 192
- Prints recommended if/else branches for C++ launch template
"""

from dataclasses import dataclass
from typing import List
import argparse
import math

@dataclass
class V100Spec:
    num_sms: int = 80
    max_threads_per_sm: int = 2048
    max_warps_per_sm: int = 64
    max_blocks_per_sm: int = 32
    max_registers_per_sm: int = 65536
    max_registers_per_thread: int = 255
    max_smem_per_sm: int = 96 * 1024
    max_smem_per_block: int = 96 * 1024
    mma_atom_m: int = 8
    mma_atom_n: int = 8
    mma_atom_k: int = 4
    memory_bandwidth_gbs: float = 900.0
    peak_tflops_fp16_tensor: float = 125.0
    l2_cache_kb: int = 6144
    threads_per_warp: int = 32

@dataclass
class FAConfig:
    head_dim: int
    block_m: int
    block_n: int
    cta_warps: int
    mma_layout_warps: int = 0

    def __post_init__(self):
        if self.mma_layout_warps <= 0:
            self.mma_layout_warps = self.cta_warps

@dataclass
class FAConfigResult:
    config: FAConfig
    spec: V100Spec
    n_threads: int = 0
    warp_rows: int = 0
    smem_q: int = 0
    smem_kv: int = 0
    smem_p: int = 0
    smem_total: int = 0
    reg_estimate: int = 0
    blocks_per_sm: int = 0
    occupancy_pct: float = 0.0
    active_warps_per_sm: int = 0
    arithmetic_intensity: float = 0.0
    roofline_bound: str = ""
    estimated_tflops: float = 0.0
    estimated_effective_bandwidth_gbs: float = 0.0
    issues: List[str] = None
    recommendations: List[str] = None

    def __post_init__(self):
        self.issues = self.issues or []
        self.recommendations = self.recommendations or []

def estimate_registers(config: FAConfig, spec: V100Spec) -> int:
    kWarpRows = config.block_m // config.cta_warps
    base_regs = 48
    o_accum_regs = kWarpRows * config.head_dim // 32
    mma_frag_regs = 32
    p_regs = min(kWarpRows * config.block_n // (4 * config.cta_warps * 32) + 4, 32)
    load_regs = config.block_n * config.head_dim // (32 * config.cta_warps) + 8
    softmax_regs = 16
    total = base_regs + o_accum_regs + mma_frag_regs + p_regs + load_regs + softmax_regs
    total = min(total, spec.max_registers_per_thread)
    return max(total, 64)

def calculate_smem(config: FAConfig):
    elem_size = 2
    q = config.block_m * config.head_dim * elem_size
    kv = config.block_n * config.head_dim * elem_size
    p = config.block_m * config.block_n * elem_size
    return q, kv, p, q + kv + p

def calculate_occupancy(config: FAConfig, spec: V100Spec, reg_estimate: int):
    n_threads = config.cta_warps * spec.threads_per_warp
    _, _, _, smem_total = calculate_smem(config)
    
    if n_threads == 0:
        return 0, 0, 0, 0, 1, 0.0, 0
    
    blocks_by_reg = spec.max_registers_per_sm // (n_threads * reg_estimate) if reg_estimate > 0 else spec.max_blocks_per_sm
    blocks_by_smem = spec.max_smem_per_sm // smem_total if smem_total > 0 else spec.max_blocks_per_sm
    blocks_by_warp = spec.max_warps_per_sm // config.cta_warps if config.cta_warps > 0 else spec.max_blocks_per_sm
    blocks_by_thread = spec.max_threads_per_sm // n_threads if n_threads > 0 else spec.max_blocks_per_sm
    blocks_per_sm = min(blocks_by_reg, blocks_by_smem, blocks_by_warp, blocks_by_thread, spec.max_blocks_per_sm)
    blocks_per_sm = max(1, blocks_per_sm)
    active_warps = blocks_per_sm * config.cta_warps
    occupancy_pct = (active_warps / spec.max_warps_per_sm) * 100.0
    return blocks_by_reg, blocks_by_smem, blocks_by_warp, blocks_by_thread, blocks_per_sm, occupancy_pct, active_warps

def analyze_config(config: FAConfig, spec: V100Spec = None) -> FAConfigResult:
    if spec is None:
        spec = V100Spec()
    res = FAConfigResult(config=config, spec=spec)
    res.n_threads = config.cta_warps * spec.threads_per_warp
    res.warp_rows = config.block_m // config.cta_warps

    # constraints
    if config.block_m % config.cta_warps != 0:
        res.issues.append(f"kBlockM must be divisible by kCtaWarps")
    if res.warp_rows not in (8, 16, 32, 64):
        res.issues.append(f"kWarpRows must be 8,16,32,64 (got {res.warp_rows})")
    if config.head_dim % 32 != 0:
        res.issues.append("kHeadDim must be multiple of 32")
    if config.block_n % spec.mma_atom_n != 0:
        res.issues.append(f"kBlockN not multiple of MMA atom N ({spec.mma_atom_n})")
    if config.block_n > config.head_dim:
        res.issues.append("kBlockN > kHeadDim (would read beyond matrix bounds)")
    
    # 已在后面实现正确的检查：
    # kMmaThreads = 8 * kMmaLayoutWarps
    # 当 kBlockN >= 32 时，要求 kMmaThreads >= 16
    
    # 新增：SM70 TiledMma 线程数检查
    # kMmaThreads = 8 * kMmaLayoutWarps (每个 MMA atom 8 线程)
    kMmaThreads = 8 * config.mma_layout_warps
    # 当 kBlockN 较大时，需要足够的 MMA 线程来访问 shared memory
    # 经验规则：kBlockN >= 32 时，kMmaThreads 应该 >= 16
    if config.block_n >= 32 and kMmaThreads < 16:
        res.issues.append(
            f"CRITICAL: kBlockN={config.block_n} requires kMmaThreads>=16, "
            f"but kMmaLayoutWarps={config.mma_layout_warps} gives only {kMmaThreads} threads. "
            f"This will cause shared memory out-of-bounds access. "
            f"Increase kMmaLayoutWarps to {config.mma_layout_warps * 2} or reduce kBlockN to 16."
        )
    
    # 检查 kGmemRowsPerThread 计算是否会除零
    kBlockKSmem = 64 if config.head_dim % 64 == 0 else 32
    kGmemElemsPerLoad = 16 // 2  # sizeof(uint128_t) / sizeof(Element)
    kGmemThreadsPerRow = kBlockKSmem // kGmemElemsPerLoad if kGmemElemsPerLoad > 0 else 1
    n_threads = config.cta_warps * spec.threads_per_warp
    kGmemRowsPerThread_denominator = n_threads // kGmemThreadsPerRow if kGmemThreadsPerRow > 0 else 1
    if kGmemRowsPerThread_denominator == 0:
        res.issues.append(
            f"kGmemRowsPerThread calculation would divide by zero: "
            f"kBlockN={config.block_n} / (kNThreads={n_threads} / kGmemThreadsPerRow={kGmemThreadsPerRow}) = 0. "
            f"Try reducing kCtaWarps or increasing kBlockN"
        )
    elif config.block_n < kGmemRowsPerThread_denominator:
        res.issues.append(
            f"kGmemRowsPerThread = {config.block_n} / {kGmemRowsPerThread_denominator} < 1. "
            f"kBlockN must be >= kNThreads/kGmemThreadsPerRow. "
            f"Try reducing kCtaWarps from {config.cta_warps} to {config.cta_warps // 2} or increasing kBlockN"
        )
    
    q, kv, p, total = calculate_smem(config)
    res.smem_q, res.smem_kv, res.smem_p, res.smem_total = q, kv, p, total
    if total > spec.max_smem_per_block:
        res.issues.append(f"SMEM {total/1024:.1f}KB exceeds 96KB")

    reg = estimate_registers(config, spec)
    res.reg_estimate = reg
    if reg >= spec.max_registers_per_thread:
        res.issues.append(f"Regs at max ({reg})")

    res.blocks_by_reg, res.blocks_by_smem, res.blocks_by_warp, res.blocks_by_thread = 0, 0, 0, 0

    res.blocks_by_reg, res.blocks_by_smem, res.blocks_by_warp, res.blocks_by_thread, \
        blk_per_sm, occ, act_warps = calculate_occupancy(config, spec, reg)
    res.blocks_per_sm = blk_per_sm
    res.occupancy_pct = occ
    res.active_warps_per_sm = act_warps

    res.arithmetic_intensity = float(config.block_m)
    ridge = spec.peak_tflops_fp16_tensor * 1000 / spec.memory_bandwidth_gbs

    if res.arithmetic_intensity > ridge:
        res.roofline_bound = "Compute-bound"
        res.estimated_tflops = spec.peak_tflops_fp16_tensor * (occ / 100.0) * 0.85
    else:
        res.roofline_bound = "Memory-bound"
        eff_bw = spec.memory_bandwidth_gbs * (occ / 100.0) * 0.85
        res.estimated_effective_bandwidth_gbs = eff_bw
        res.estimated_tflops = eff_bw * res.arithmetic_intensity / 1000.0

    # recommendations
    if occ < 20.0 and config.cta_warps == 8:
        res.recommendations.append("Switch to 4 warps to reduce register pressure and increase occupancy")
    if total > 48 * 1024 and blk_per_sm == 1:
        res.recommendations.append("SMEM >48KB limits to 1 block/SM; consider smaller tiles for 2 blocks/SM")

    # 新增：SM70 MMA 布局硬约束
    # 每个 warp 至少需要 8 个 N 列才能完整执行一次 m8n8k4
    n_per_warp = config.block_n // config.mma_layout_warps
    if n_per_warp < 8:
        res.issues.append(
            f"kBlockN/kMmaLayoutWarps = {n_per_warp} < 8. "
            f"Each warp needs at least 8 N-columns for SM70 m8n8k4 MMA atom, can try -mw {1 if n_per_warp <= 4 else 4}"
        )

    return res

def generate_candidate_configs(head_dim: int, cta_warps_list=(2, 4, 8)) -> List[FAConfig]:
    configs = []
    for cw in cta_warps_list:
        for wr in [8, 16, 32, 64]:
            bm = wr * cw
            if bm > 256:
                continue
            for bn in [32, 64, 128]:
                if bn > head_dim:
                    continue
                if cw == 2:
                    configs.append(FAConfig(head_dim, bm, bn, cw, mma_layout_warps=1))
                elif cw == 8:
                    # 同时生成 mma_layout_warps=8 (默认) 和 =4 (减少每个 warp 的 N 列数)
                    configs.append(FAConfig(head_dim, bm, bn, cw, mma_layout_warps=8))
                    # 只有当 bn//4 >= 8 时，mma_layout_warps=4 才合法 (即 bn>=32)
                    if bn // 4 >= 8:
                        configs.append(FAConfig(head_dim, bm, bn, cw, mma_layout_warps=4))
                else:  # cw == 4
                    configs.append(FAConfig(head_dim, bm, bn, cw))
    return configs

def has_hard_error(result: FAConfigResult) -> bool:
    for issue in result.issues:
        if any(kw in issue for kw in [
            "divisible",
            "kWarpRows must be",
            "multiple of 32",
            "MMA atom N",
            "beyond matrix bounds",
            "kBlockN/kMmaLayoutWarps",
            "exceeds 96KB",          # 新增：共享内存超限
            "CRITICAL:",             # 新增：严重错误（如 shared memory 越界）
        ]):
            return True
    return False

def rank_by_estimated_tflops(configs: List[FAConfig], spec: V100Spec = None):
    if spec is None:
        spec = V100Spec()
    results = []
    for cfg in configs:
        res = analyze_config(cfg, spec)
        if has_hard_error(res):    # 只跳过硬错误
            continue
        # 软约束（reg 超 255、smem 超 96KB）保留，仍参与排名
        results.append((res, res.estimated_tflops))
    results.sort(key=lambda x: -x[1])
    return results

def calculate_grid(seqlen_q: int, block_m: int, batch_size: int = 1, num_heads: int = 1):
    num_m_block = math.ceil(seqlen_q / block_m)
    return num_m_block, batch_size, num_heads

def print_result(result: FAConfigResult, score=None, seqlen_q=None, batch_size=1, num_heads=1):
    c = result.config
    print(f"  Config: kHeadDim={c.head_dim}, kBlockM={c.block_m}, kBlockN={c.block_n}, "
          f"kWarpCount={c.cta_warps}, kMmaLayoutWarps={c.mma_layout_warps}")
    print(f"  Derived: {result.n_threads} threads, kWarpRows={result.warp_rows}")
    print(f"  SMEM: Q={result.smem_q/1024:.1f}K KV={result.smem_kv/1024:.1f}K P={result.smem_p/1024:.1f}K "
          f"Total={result.smem_total/1024:.1f}K")
    print(f"  Registers/Block: {result.reg_estimate * result.n_threads}  "
          f"({result.reg_estimate * result.n_threads / result.spec.max_registers_per_sm * 100:.1f}% of 65536/SM)"
          f" Regs/thread: {result.reg_estimate} ")
    print(f"  Blocks/SM: {result.blocks_per_sm} | Registers: {result.blocks_by_reg} blocks/SM | SMEM: {result.blocks_by_smem} blocks/SM |"
          f"  Warps: {result.blocks_by_warp} blocks/SM | Threads: {result.blocks_by_thread} blocks/SM")
    print(f"  Est. TFlops: {result.estimated_tflops:.1f} | Occupancy: {result.occupancy_pct:.1f}% "
          f"| AI: {result.arithmetic_intensity:.0f} FLOPS/byte | Bottleneck: {result.roofline_bound}")

    if seqlen_q is not None:
        grid_x, grid_y, grid_z = calculate_grid(seqlen_q, c.block_m, batch_size, num_heads)
        total_blocks = grid_x * grid_y * grid_z
        print(f"  Grid: ({grid_x}, {grid_y}, {grid_z}) | Total Blocks: {total_blocks} "
              f"(batch={batch_size}, seqlen_q={seqlen_q}, heads={num_heads}, dim={c.head_dim})")

    if result.recommendations:
        for r in result.recommendations:
            print(f"  -> {r}")
    if result.issues:
        for i in result.issues:
            print(f"  !! {i}")


def evaluate_config_for_seqlen(config: FAConfig, seqlen: int, spec: V100Spec) -> float:
    res = analyze_config(config, spec)
    if has_hard_error(res):
        return 0.0

    base_tflops = res.estimated_tflops

    # KV 迭代开销
    kv_iters = math.ceil(seqlen / config.block_n)
    iter_efficiency = 1.0 / (1.0 + 0.002 * kv_iters)

    # M‑blocks 并行度惩罚（加强对短序列的惩罚）
    m_blocks = math.ceil(seqlen / config.block_m)
    # 原为线性，现改为平方根，使小 m_blocks 的惩罚更明显
    parallel_eff = min(1.0, (m_blocks / spec.num_sms) ** 0.5)

    return base_tflops * iter_efficiency * parallel_eff

def _select_best_config(candidates, spec, prefer_high_ai=True, prefer_occupancy=False):
    if not candidates:
        return None
    if prefer_high_ai and prefer_occupancy:
        return max(candidates, key=lambda c: (
            analyze_config(c, spec).estimated_tflops *
            analyze_config(c, spec).blocks_per_sm
        ))
    elif prefer_high_ai:
        return max(candidates, key=lambda c: analyze_config(c, spec).estimated_tflops)
    elif prefer_occupancy:
        return max(candidates, key=lambda c: (
            analyze_config(c, spec).occupancy_pct * analyze_config(c, spec).blocks_per_sm
        ))
    else:
        return max(candidates, key=lambda c: analyze_config(c, spec).estimated_tflops)


def _fmt_kernel(hdim, cfg):
    return f"Flash_fwd_kernel_traits<{hdim}, {cfg.block_m}, {cfg.block_n}, {cfg.cta_warps}, {cfg.mma_layout_warps}>"


def generate_dispatch_table(head_dims, seqlen_ranges, spec=None):
    if spec is None:
        spec = V100Spec()
    sorted_ranges = sorted(seqlen_ranges, reverse=True)

    print("// Auto-generated dispatch logic for V100 SM70")
    print("// Based on seqlen-aware configuration selection\n")

    for hdim in head_dims:
        print(f"// ----- HeadDim={hdim} -----")
        all_candidates = generate_candidate_configs(hdim)
        valid = [c for c in all_candidates if not has_hard_error(analyze_config(c, spec))]

        best_configs = []
        for threshold in sorted_ranges:
            if threshold >= 2048:
                candidates_subset = [c for c in valid if c.block_m >= 128]
                if not candidates_subset:
                    candidates_subset = valid
                best_cfg = max(candidates_subset, key=lambda c: analyze_config(c, spec).estimated_tflops)
            elif threshold >= 1024:
                best_cfg = max(valid, key=lambda c: (
                    analyze_config(c, spec).occupancy_pct *
                    analyze_config(c, spec).arithmetic_intensity
                ))
            else:
                best_cfg = max(valid, key=lambda c: (
                    analyze_config(c, spec).occupancy_pct -
                    0.1 * (threshold // c.block_n)
                ))
            best_configs.append((threshold, best_cfg))

        for idx, (threshold, cfg) in enumerate(best_configs):
            if idx == 0:
                print(f"    // hdim={hdim} seqlen_q >= {threshold} ")
                print(f"    if (params.seqlen_q >= {threshold}) {{")
                print(f"        run_flash_fwd<{_fmt_kernel(hdim, cfg)}, Is_dropout, Is_causal>(params, stream);")
            else:
                prev_threshold = best_configs[idx-1][0]
                print(f"    // hdim={hdim} {prev_threshold} > seqlen_q >= {threshold}")
                print(f"    }} else if (params.seqlen_q >= {threshold}) {{")
                print(f"        run_flash_fwd<{_fmt_kernel(hdim, cfg)}, Is_dropout, Is_causal>(params, stream);")

        last_threshold = best_configs[-1][0]
        last_cfg = best_configs[-1][1]
        print(f"    // hdim={hdim} seqlen_q < {last_threshold}")
        print(f"    }} else {{")
        print(f"        run_flash_fwd<{_fmt_kernel(hdim, last_cfg)}, Is_dropout, Is_causal>(params, stream);")
        print(f"    }}")
        print()


def generate_cross_attn_dispatch(head_dims, spec=None):
    if spec is None:
        spec = V100Spec()

    KV_THRESHOLDS = [256, 1024]
    Q_LONG_THRESHOLD = 2048

    print("// Auto-generated cross-attention-aware dispatch logic for V100 SM70")
    print("// Considers both seqlen_q and seqlen_k for optimal kernel selection\n")

    for hdim in head_dims:
        print(f"// ----- HeadDim={hdim} -----")
        all_candidates = generate_candidate_configs(hdim)
        valid = [c for c in all_candidates if not has_hard_error(analyze_config(c, spec))]

        high_ai_cfgs = [c for c in valid if c.block_m >= 128 and c.cta_warps >= 8]
        if not high_ai_cfgs:
            high_ai_cfgs = [c for c in valid if c.block_m >= 128]
        med_cfgs = [c for c in valid if 64 <= c.block_m < 128 and c.cta_warps == 4]
        if not med_cfgs:
            med_cfgs = [c for c in valid if c.block_m >= 64 and c.cta_warps == 4]
        small_cfgs = [c for c in valid if c.block_m <= 64 and c.cta_warps == 4]
        if not small_cfgs:
            small_cfgs = [c for c in valid if c.block_m <= 64]

        cfg_long_q_long_kv = _select_best_config(high_ai_cfgs, spec, prefer_high_ai=True)
        cfg_long_q_med_kv = _select_best_config(
            [c for c in med_cfgs if c.block_n >= 32], spec, prefer_high_ai=True, prefer_occupancy=True)
        cfg_long_q_short_kv = _select_best_config(
            [c for c in small_cfgs if c.block_n <= 64], spec, prefer_occupancy=True)
        cfg_short_q_long_kv = _select_best_config(
            [c for c in med_cfgs if c.block_n >= 32], spec, prefer_high_ai=True, prefer_occupancy=True)
        cfg_short_q_med_kv = _select_best_config(
            [c for c in med_cfgs if c.block_n <= 64], spec, prefer_occupancy=True)
        cfg_short_q_short_kv = _select_best_config(
            [c for c in small_cfgs if c.block_n <= 64], spec, prefer_occupancy=True)

        if not all([cfg_long_q_long_kv, cfg_long_q_med_kv, cfg_long_q_short_kv,
                     cfg_short_q_long_kv, cfg_short_q_med_kv, cfg_short_q_short_kv]):
            print(f"    // WARNING: Could not find all configs for hdim={hdim}, falling back to simple dispatch")
            generate_dispatch_table([hdim], [2048, 1024, 512], spec)
            continue

        print(f"    if (params.seqlen_q >= {Q_LONG_THRESHOLD}) {{")
        print(f"        if (params.seqlen_k <= {KV_THRESHOLDS[0]}) {{")
        print(f"            run_flash_fwd<{_fmt_kernel(hdim, cfg_long_q_short_kv)}, Is_dropout, Is_causal>(params, stream);")
        print(f"        }} else if (params.seqlen_k <= {KV_THRESHOLDS[1]}) {{")
        print(f"            run_flash_fwd<{_fmt_kernel(hdim, cfg_long_q_med_kv)}, Is_dropout, Is_causal>(params, stream);")
        print(f"        }} else {{")
        print(f"            run_flash_fwd<{_fmt_kernel(hdim, cfg_long_q_long_kv)}, Is_dropout, Is_causal>(params, stream);")
        print(f"        }}")
        print(f"    }} else {{")
        print(f"        if (params.seqlen_k <= {KV_THRESHOLDS[0]}) {{")
        print(f"            run_flash_fwd<{_fmt_kernel(hdim, cfg_short_q_short_kv)}, Is_dropout, Is_causal>(params, stream);")
        print(f"        }} else if (params.seqlen_k <= {KV_THRESHOLDS[1]}) {{")
        print(f"            run_flash_fwd<{_fmt_kernel(hdim, cfg_short_q_med_kv)}, Is_dropout, Is_causal>(params, stream);")
        print(f"        }} else {{")
        print(f"            run_flash_fwd<{_fmt_kernel(hdim, cfg_short_q_long_kv)}, Is_dropout, Is_causal>(params, stream);")
        print(f"        }}")
        print(f"    }}")
        print()

def main():
    parser = argparse.ArgumentParser(description="V100 SM70 FA config optimizer")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for analysis")
    parser.add_argument("--num_heads", type=int, default=10, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, nargs="+", default=[128],
                        help="Head dimensions to analyze (e.g. 64 96 128 )")
    parser.add_argument("--seqlen_ranges", type=int, nargs="+", default=[2048, 1024, 512],
                        help="Sequence length thresholds for dispatch generation (descending)")

    parser.add_argument("--all", action="store_true", help="Print all valid configs, not just top ones")
    parser.add_argument("--dispatch_only", action="store_true", help="Only print dispatch code")
    parser.add_argument("--cross_attn", action="store_true", help="Generate cross-attention-aware dispatch (seqlen_q + seqlen_k)")

    parser.add_argument("--block_m", type=int, default=None,
                        help="Analyze a single specific block_m (overrides ranking)")
    parser.add_argument("--block_n", type=int, default=64,
                        help="block_n when analyzing a single config")
    parser.add_argument("--warps", type=int, default=8,
                        help="Number of warps when analyzing a single config")

    parser.add_argument("-mw", "--mma_layout_warps", type=int, default=0,
                        help="kMmaLayoutWarps (default: auto-deducted)")
    args = parser.parse_args()

    spec = V100Spec()

    # ---------- 单配置分析模式 ----------
    if args.block_m is not None:
        hdim = args.head_dim[0]                     # 只使用第一个 head_dim
        mma_layout = args.mma_layout_warps
        if mma_layout <= 0:                         # 自动推导 kMmaLayoutWarps
            if args.warps == 2:
                mma_layout = 1                      # 2‑warps 必须为 1
            else:
                mma_layout = args.warps             # 其它情况默认等于 warps
        config = FAConfig(hdim, args.block_m, args.block_n, args.warps, mma_layout)
        result = analyze_config(config, spec)
        print(f"Single Config Analysis for HeadDim={hdim}:")
        print_result(result, seqlen_q=args.seqlen_ranges[0], batch_size=args.batch_size, num_heads=args.num_heads)
        return                                      # 结束，不再执行后续排名和调度

    # ---------- 原有多配置排名与调度模式 ----------
    if not args.dispatch_only:
        for hdim in args.head_dim:
            print(f"\n{'='*60}")
            print(f"  HeadDim={hdim} - All Valid Configs Ranked by Est. TFLOPS")
            print(f"{'='*60}")
            configs = generate_candidate_configs(hdim)
            ranked = rank_by_estimated_tflops(configs, spec)
            top_n = len(ranked) if args.all else 5
            for i, (res, score) in enumerate(ranked[:top_n]):
                print(f"--- Rank #{i+1} (est. {score:.2f} TFlops) ---")
                print_result(res)
                print()

    generate_dispatch_table(args.head_dim, args.seqlen_ranges, spec)

    if args.cross_attn:
        print("// Cross-Attention-Aware Dispatch (seqlen_q + seqlen_k)")
        generate_cross_attn_dispatch(args.head_dim, spec)

if __name__ == "__main__":
    main()


# python tools/sm70_cfg.py --head_dim 128 --block_m 128 --block_n 64 --warps 4
# python tools/sm70_cfg.py --head_dim 128 --seqlen_ranges 2048 1024

