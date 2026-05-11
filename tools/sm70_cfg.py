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
    blocks_by_reg = spec.max_registers_per_sm // (n_threads * reg_estimate)
    blocks_by_smem = spec.max_smem_per_sm // smem_total if smem_total > 0 else spec.max_blocks_per_sm
    blocks_by_warp = spec.max_warps_per_sm // config.cta_warps
    blocks_by_thread = spec.max_threads_per_sm // n_threads
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
    
    # 新增：SM70 要求 kMmaThreads == 32，而 kMmaThreads = 32 * kMmaLayoutWarps
    if config.mma_layout_warps != 1 and config.cta_warps == 2:
        res.issues.append(
            "kCtaWarps=2 must have kMmaLayoutWarps=1 to keep kMmaThreads=32"
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
            "Each warp needs at least 8 N-columns for SM70 m8n8k4 MMA atom, can try --mma_layout_warps=4"
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
            "exceeds 96KB"          # 新增：共享内存超限
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

def print_result(result: FAConfigResult, score=None):
    c = result.config
    print(f"  Config: kHeadDim={c.head_dim}, kBlockM={c.block_m}, kBlockN={c.block_n}, "
          f"kWarpCount={c.cta_warps}, kMmaLayoutWarps={c.mma_layout_warps}")
    print(f"  Derived: {result.n_threads} threads, kWarpRows={result.warp_rows}")
    print(f"  SMEM: Q={result.smem_q/1024:.1f}K KV={result.smem_kv/1024:.1f}K P={result.smem_p/1024:.1f}K "
          f"Total={result.smem_total/1024:.1f}K")
    print(f"  Blocks/SM: {result.blocks_per_sm}  Regs/thread: {result.reg_estimate}  "
          f"Occupancy: {result.occupancy_pct:.1f}%")
    print(f"  Registers: {result.blocks_by_reg} blocks/SM | SMEM: {result.blocks_by_smem} blocks/SM |"
          f" Warps: {result.blocks_by_warp} blocks/SM | Threads: {result.blocks_by_thread} blocks/SM")
    print(f"  AI: {result.arithmetic_intensity:.0f} FLOPS/byte  Bottleneck: {result.roofline_bound}")
    print(f"  Est. TFlops: {result.estimated_tflops:.1f}")
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
            if threshold >= 2048:   # 长序列：优先大 kBlockM（高算术强度）
                # 只考虑 block_m >= 128 的配置，在其中选 estimated_tflops 最高
                candidates_subset = [c for c in valid if c.block_m >= 128]
                if not candidates_subset:
                    candidates_subset = valid
                best_cfg = max(candidates_subset, key=lambda c: analyze_config(c, spec).estimated_tflops)
            elif threshold >= 1024: # 中等序列：兼顾算术强度与占用率
                best_cfg = max(valid, key=lambda c: (
                    analyze_config(c, spec).occupancy_pct *
                    analyze_config(c, spec).arithmetic_intensity
                ))
            else:                   # 短序列：更看重占用率，以及更少的 KV 迭代次数
                best_cfg = max(valid, key=lambda c: (
                    analyze_config(c, spec).occupancy_pct -
                    0.1 * (threshold // c.block_n)  # 轻微惩罚高 KV 迭代次数
                ))
            best_configs.append((threshold, best_cfg))

        # 输出 if/else 块
        for idx, (threshold, cfg) in enumerate(best_configs):
            mma_layout = cfg.mma_layout_warps  # 直接使用配置中的值
            if idx == 0:
                print(f"    // hdim={hdim} seqlen_q >= {threshold} ")
                print(f"    if (params.seqlen_q >= {threshold}) {{")
                print(f"        run_flash_fwd<Flash_fwd_kernel_traits<{hdim}, {cfg.block_m}, {cfg.block_n}, {cfg.cta_warps}, {mma_layout}>, Is_dropout, Is_causal>(params, stream);")
            else:
                prev_threshold = best_configs[idx-1][0]
                print(f"    // hdim={hdim} {prev_threshold} > seqlen_q >= {threshold}")
                print(f"    }} else if (params.seqlen_q >= {threshold}) {{")
                print(f"        run_flash_fwd<Flash_fwd_kernel_traits<{hdim}, {cfg.block_m}, {cfg.block_n}, {cfg.cta_warps}, {mma_layout}>, Is_dropout, Is_causal>(params, stream);")

        last_threshold = best_configs[-1][0]
        last_cfg = best_configs[-1][1]
        mma_layout_last = last_cfg.mma_layout_warps
        print(f"    // hdim={hdim} seqlen_q < {last_threshold}")
        print(f"    }} else {{")
        print(f"        run_flash_fwd<Flash_fwd_kernel_traits<{hdim}, {last_cfg.block_m}, {last_cfg.block_n}, {last_cfg.cta_warps}, {mma_layout_last}>, Is_dropout, Is_causal>(params, stream);")
        print(f"    }}")
        print()

def main():
    parser = argparse.ArgumentParser(description="V100 SM70 FA config optimizer")
    parser.add_argument("--head_dim", type=int, nargs="+", default=[128],
                        help="Head dimensions to analyze (e.g. 64 96 128 )")
    parser.add_argument("--seqlen_ranges", type=int, nargs="+", default=[2048, 1024, 512],
                        help="Sequence length thresholds for dispatch generation (descending)")

    parser.add_argument("--show_all", action="store_true", help="Print all valid configs, not just top ones")
    parser.add_argument("--dispatch_only", action="store_true", help="Only print dispatch code")

    parser.add_argument("--block_m", type=int, default=None,
                        help="Analyze a single specific block_m (overrides ranking)")
    parser.add_argument("--block_n", type=int, default=64,
                        help="block_n when analyzing a single config")
    parser.add_argument("--warps", type=int, default=8,
                        help="Number of warps when analyzing a single config")

    parser.add_argument("--mma_layout_warps", type=int, default=0,
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
        print_result(result)
        return                                      # 结束，不再执行后续排名和调度

    # ---------- 原有多配置排名与调度模式 ----------
    if not args.dispatch_only:
        for hdim in args.head_dim:
            print(f"\n{'='*60}")
            print(f"  HeadDim={hdim} - All Valid Configs Ranked by Est. TFLOPS")
            print(f"{'='*60}")
            configs = generate_candidate_configs(hdim)
            ranked = rank_by_estimated_tflops(configs, spec)
            top_n = len(ranked) if args.show_all else 5
            for i, (res, score) in enumerate(ranked[:top_n]):
                print(f"--- Rank #{i+1} (est. {score:.2f} TFlops) ---")
                print_result(res)
                print()

    generate_dispatch_table(args.head_dim, args.seqlen_ranges, spec)

if __name__ == "__main__":
    main()


# python tools/sm70_cfg.py --head_dim 128 --block_m 128 --block_n 64 --warps 4
# python tools/sm70_cfg.py --head_dim 128 --seqlen_ranges 2048 1024

