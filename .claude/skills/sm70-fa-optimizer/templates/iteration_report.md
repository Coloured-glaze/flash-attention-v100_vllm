# Iteration {{iter}} — SM70 FA Forward Kernel Analysis

**Kernel profiled (input)**: `{{best_file_before}}`
**Time before**: {{best_ms_before}} ms
**GPU / arch**: {{gpu_name}} / SM70 (V100)
**Dimensions**: B={{batch}}, H={{nheads}}, S={{seqlen}}, D={{headdim}}, causal={{causal}}

---

## Roofline Analysis (from `roofline.json`)

| Axis | Δ (gap) | Utilization | Budget |
|------|---------|-------------|--------|
| Compute | {{delta_compute}} | {{compute_util_pct}}% | {{budget_compute}} |
| Memory | {{delta_memory}} | {{memory_util_pct}}% | {{budget_memory}} |
| Latency | {{delta_latency}} | max stall {{max_stall_pct}}% | {{budget_latency}} |

**Primary bound**: {{bound}}

**FA-specific metrics**:
| Metric | Value | Status |
|--------|-------|--------|
| Bank conflict (load) | {{bank_conflict_load}}% | {{bc_load_status}} |
| Bank conflict (store) | {{bank_conflict_store}}% | {{bc_store_status}} |
| Occupancy | {{occupancy}}% | {{occupancy_status}} |
| Registers/thread | {{regs_per_thread}} | {{reg_status}} |
| MIO throttle | {{mio_throttle}}% | {{mio_status}} |

---

## Current Tile Configuration

| Parameter | Value |
|-----------|-------|
| kHeadDim | {{headdim}} |
| kBlockM | {{block_m}} |
| kBlockN | {{block_n}} |
| kCtaWarps | {{cta_warps}} |
| kMmaLayoutWarps | {{mma_layout_warps}} |
| kWarpRows | {{warp_rows}} |
| kNThreads | {{nthreads}} |
| kBlockKSmem | {{block_k_smem}} |
| kSwizzle | {{swizzle}} |
| kSmemSize | {{smem_size_kb}} KB |

---

## Top ncu metrics (from `ncu_top.json`)

### Stall Reasons
{{stall_metrics_table}}

### Memory Hierarchy
{{memory_metrics_table}}

### Compute / Instructions
{{compute_metrics_table}}

---

## Diagnosis

_Which axis is the dominant bottleneck right now? Cite specific metric values. Consider the SM70-specific context (no async copy, 96KB smem, 8×8×4 TC)._

> **Current baseline profile analysis:**
> - Memory-bound (89.77% memory throughput) with MIO throttle as the #1 stall (46.3%)
> - Register pressure limits occupancy to 12.5% (231 regs/thread → 2 blocks/SM max)
> - Shared memory bank conflicts at 53.1% (load) and 17.5% (store) → major bottleneck on MIO pipe
> - Only 0.50 eligible warps per scheduler → compute units starved despite 89% memory utilization
> - This is a **shared-memory-pipe-bound** kernel, not DRAM-bound (only 1.15% DRAM throughput)

---

## Chosen methods

For each, state: (a) **priority level** from `sm70_fa_catalog.md`, (b) metric evidence, (c) the method, (d) the implementation delta (which kernel_traits.h parameter or code change), (e) expected ncu metric shift.

### {{axis_1}} — `{{method_1_id}}` (Priority: {{P_level_1}})
- **Budget for this axis**: {{budget_axis_1}}
- **Priority scan**: _List all higher-priority methods that were scanned and why each was skipped_
- **Trigger evidence**: _(cite specific ncu metric name = value)_
- **Trigger strength**: _(continuous value 0-1 if b_axis ≥ 2)_
- **Method**: _What change? (e.g., "Reduce kWarpRows from 32 to 16 by changing kBlockM from 128 to 64")_
- **Delta vs current best**: _Which trait params change? New values?_
- **Expected ncu shift**: _e.g., "regs/thread: 231→~150; occupancy: 12.5%→~18.75%; MIO throttle: 46%→~38%"_
- **Risks / coupling**: _e.g., "Smaller kWarpRows → more loop iterations → more sync → barrier stall may increase"_
- **SM70 constraints check**: _e.g., "kWarpRows=16 ∈ {8,16,32,64} ✅; kSmemSize 32KB ≤ 96KB ✅"_

### {{axis_2}} — `{{method_2_id}}` (Priority: {{P_level_2}})
- **Budget for this axis**: {{budget_axis_2}}
- **Priority scan**: _List higher-priority methods scanned + skip reasons_
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:
- **SM70 constraints check**:

### {{axis_3}} — `{{method_3_id}}` (Priority: {{P_level_3}})
_(only if B=3 methods; omit if axis budget is 0)_
- **Budget for this axis**: {{budget_axis_3}}
- **Priority scan**:
- **Trigger evidence**:
- **Method**:
- **Delta vs current best**:
- **Expected ncu shift**:
- **Risks / coupling**:
- **SM70 constraints check**:

---

## Orthogonality check

_Verify: (1) no pair is the same optimization under two names, (2) methods don't conflict (e.g., tile size increase + register pressure reduction can conflict), (3) all SM70-compatible, (4) axis distribution matches roofline budget._

---

## Excluded candidates (higher-priority methods that were skipped)

_List every higher-priority method on each axis that was NOT selected, with the exact reason:_

- `memory.bank_conflict_swizzle` (P1) — skipped: {{reason or "selected"}}
- `compute.reduce_register_pressure` (P1) — skipped: {{reason or "selected"}}
- `latency.mio_throttle_reduce` (P1) — skipped: {{reason or "selected"}}

---

## Branch variants

_Describe the K hyperparameter variants generated for branch-and-select:_

| Branch | kBlockM | kBlockN | kCtaWarps | kWarpRows | kBlockKSmem | kSwizzle | Expected Δ |
|--------|---------|---------|-----------|-----------|-------------|----------|------------|
| b1 | {{b1_m}} | {{b1_n}} | {{b1_w}} | {{b1_wr}} | {{b1_ks}} | {{b1_sw}} | — |
| b2 | {{b2_m}} | {{b2_n}} | {{b2_w}} | {{b2_wr}} | {{b2_ks}} | {{b2_sw}} | — |
| b3 | {{b3_m}} | {{b3_n}} | {{b3_w}} | {{b3_wr}} | {{b3_ks}} | {{b3_sw}} | — |
| b4 | {{b4_m}} | {{b4_n}} | {{b4_w}} | {{b4_wr}} | {{b4_ks}} | {{b4_sw}} | — |

---

## Result (filled after benchmarking)

| Branch | Time (ms) | vs Baseline | Correctness | Selected |
|--------|-----------|-------------|-------------|----------|
| b1 | {{b1_time}} | {{b1_vs_base}}% | {{b1_correct}} | {{b1_selected}} |
| b2 | {{b2_time}} | {{b2_vs_base}}% | {{b2_correct}} | {{b2_selected}} |
| b3 | {{b3_time}} | {{b3_vs_base}}% | {{b3_correct}} | {{b3_selected}} |
| b4 | {{b4_time}} | {{b4_vs_base}}% | {{b4_correct}} | {{b4_selected}} |

**Champion**: b{{champion}} ({{champion_time}} ms, {{champion_gain}}% faster)

---

## Ablation (from `attribution.json`)

| Method | Without (ms) | Attribution (ms) | Effective? |
|--------|-------------|-----------------|------------|
| {{method_1_id}} | {{m1_abl}} | {{m1_attr}} | {{m1_eff}} |
| {{method_2_id}} | {{m2_abl}} | {{m2_attr}} | {{m2_eff}} |
| {{method_3_id}} | {{m3_abl}} | {{m3_attr}} | {{m3_eff}} |

---

## SASS Verification (from `sass_check.json`)

| Method | Expected Pattern | Found? | Notes |
|--------|-----------------|--------|-------|
| {{method_1_id}} | {{m1_sass}} | {{m1_found}} | {{m1_sass_notes}} |
| {{method_2_id}} | {{m2_sass}} | {{m2_found}} | {{m2_sass_notes}} |
| {{method_3_id}} | {{m3_sass}} | {{m3_found}} | {{m3_sass_notes}} |

---

## Iteration Summary

**Starting**: {{best_ms_before}} ms ({{occupancy_before}}% occupancy, {{regs_before}} regs/thread, {{bc_before}}% bank conflict)
**Result**: {{new_best_ms}} ms ({{occupancy_after}}% occupancy, {{regs_after}} regs/thread, {{bc_after}}% bank conflict)
**Improvement**: {{improvement_pct}}% ({{speedup}}×)
**Methods effective**: {{effective_list}}
**Methods ineffective**: {{ineffective_list}}
**New bottlenecks**: {{new_bottlenecks}}
