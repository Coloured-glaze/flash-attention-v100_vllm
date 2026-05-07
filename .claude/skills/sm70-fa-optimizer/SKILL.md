---
name: sm70-fa-optimizer
description: Iteratively optimize the SM70 (V100) Flash Attention forward kernel using ncu-guided roofline reasoning. Use this skill when the user asks to optimize, speed up, or analyze the flash_fwd_kernel for V100 (SM70/Volta architecture), especially when they provide ncu profile data, mention "bank conflict", "occupancy", "register pressure", "shared memory", "V100", "sm70", or ask Claude to "make flash attention faster on V100". This skill is specialized for the vllm_flash_attn SM70 port and understands Volta-specific hardware constraints (Shared memory 96KB, 65536 registers/SM, 128 threads/block maximum practical, 8x8x4 Tensor Core MMA).
---

# SM70 Flash Attention Forward Kernel Optimizer (Roofline-Driven)

## What this skill does

Given the **vllm_flash_attn** project — a Flash Attention forward kernel ported to V100 (SM70/Volta) — this skill runs a **roofline-guided, iterative optimization loop** targeting the SM70-specific hardware bottlenecks:

- **SM70 constraints**: 96 KB shared memory per SM, 65536 registers per SM, max 64 warps/SM, 8x8x4 TC MMA only (no m16n8k8 like Ampere)
- **Key bottleneck patterns**: extreme register pressure (200+ regs/thread → 12.5% occupancy), bank conflicts on shared memory swizzle layouts, MIO throttle on shared memory instruction queue, low L1 hit rate from streaming global loads

The skill drives a multi-iteration loop: profile with ncu → compute roofline gaps → allocate axis budgets → pick methods from FA-specific catalog → generate K branch kernels → validate + benchmark → select champion → ablation attribution → SASS verification.

## Project-specific context

### Key files
| File | Purpose |
|------|---------|
| `csrc/flash_attn/src/flash_fwd_kernel.h` | Forward kernel main implementation |
| `csrc/flash_attn/src/kernel_traits.h` | TiledMMA definition, SM70 copy atoms, layout traits |
| `csrc/flash_attn/src/utils.h` | `convert_layout_C_to_A_v2` (SM70 register C→A conversion), copy helpers |
| `csrc/flash_attn/src/softmax.h` | Online softmax with SM70-aware fragment layouts |
| `flash_attn/flash_attn_interface.py` | Python interface for `flash_attn_func` |
| `tests/ncu_analyse/profile.sh` | Benchmark + ncu profiling script |
| `tests/ncu_analyse/test_vllm_flash_attn.py` | Python benchmark harness |

### CUTLASS/CuTe setup
- Uses **CuTe** tensor abstractions (`cute/tensor.hpp`, `cute/atom/`)
- `MMA_Atom_Arch = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>` — Volta Tensor Core MMA
- Custom 128-bit global load/store atoms: `SM70_LDG_GLOBAL_CG_128b` / `SM70_STG_GLOBAL_CG_128b`
- Fragment layout conversion: `convert_layout_C_to_A_v2()` maps TC output (C-fragment) to TC input A layout via warp shuffle — **this is a key SM70-specific optimization point**
- SASS translation: `asm("ld.global.cg.v4.u32 ...")` → `LDG.E.128.STRONG.GPU`

### benchmark & profile workflow (from profile.sh)
```bash
# benchmark
python test_vllm_flash_attn.py --flops --profile --use_sdpa

# ncu profile (requires root or --access=all for full counters)
ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:'.*fwd.*' \
    -o "profile_out" \
    python test_vllm_flash_attn.py --flops
```

### Build
```bash
cd /path/to/flash-attention-v100_vllm
bash build.sh
```
**CRITICAL**: Building is very time-consuming. Never terminate a build task. Never run build without explicit user request.

### Hardware spec (V100 SM70)
| Parameter | Value |
|-----------|-------|
| SM count | 80 |
| Shared memory / SM | 96 KB (configurable up to 96 KB) |
| Registers / SM | 65536 |
| Max threads / SM | 2048 |
| Max warps / SM | 64 |
| Max blocks / SM | 32 |
| Tensor Core | 8×8×4 FP16 (HMMA.884) |
| Global memory bandwidth | ~900 GB/s (HBM2) |
| L1 cache | 128 KB combined with shared mem |
| L2 cache | 6144 KB |
| Peak FP16 TC FLOPS | 112 TFLOPS |

### Known bottleneck profile (from reference ncu run, d=128, block=128×64)
| Metric | Value | Interpretation |
|--------|-------|----------------|
| Occupancy | 12.48% | Limited by 231 regs/thread + 98.3 KB smem |
| SM Throughput | 43.37% | Underutilized due to low occupancy |
| Memory Throughput | 89.77% | Near memory-bound |
| L1 hit rate | 0.54% | Streaming access pattern expected |
| L2 hit rate | 97.41% | Good L2 residency |
| Bank conflicts (load) | 53.1% (2.1-way) | Critical — shared load swizzle issue |
| Bank conflicts (store) | 17.5% (4.6-way) | Shared store access pattern issue |
| IPC | 1.47 | Reasonable but below TC peak due to stalls |
| Eligible warps / sched | 0.50 | Only 25% of active warps ready → stall dominated |
| MIO throttle stall | 46.3% | Shared memory instruction queue saturation |
| DRAM throughput | 1.15% | Not DRAM-bound — compute/smem bound |

### SM70 vs Ampere/Hopper key differences
| Feature | SM70 (V100) | SM80 (A100) | SM90 (H100) |
|---------|-------------|-------------|-------------|
| MMA atom | 8×8×4 | 16×8×8 / 16×8×16 | WGMMA m64nNk16 |
| Shared mem | 96 KB | 164 KB | 228 KB |
| Regs/SM | 65536 | 65536 | 65536 |
| Async copy | ❌ | ✅ cp.async | ✅ TMA |
| L2 cache | 6 MB | 40 MB | 50 MB |
| Max smem banks | 32 | 32 | 32 |
| ldmatrix | ❌ | ✅ | ✅ |
| stmatrix | ❌ | ✅ | ✅ |

---

## The optimization loop

```
0. check_env          → env.json (GPU, nvcc, CUTLASS, ncu)
1. init run folder    → run_YYYYMMDD_HHMMSS/
2. copy baseline      → baseline/ + bench once to seed best
3. for i in 1..N:
     a. profile best_kernel with ncu (--set full)  → iterv{i}/best_input.ncu-rep
     b. extract top compute/mem/latency            → ncu_top.json
     c. roofline.py: compute Δ_c, Δ_m, Δ_l        → roofline.json + axis_budget
        if near_peak (all Δ < 0.15) → early stop
     d. Claude picks methods (b_axis per axis, cap=2) → analysis.md (CoT)
     e. Claude writes K branch kernels (same methods, diff hyperparams)
     f. branch_explore.py: compile + bench all K   → select champion
     g. if champion FAIL: regenerate (max 3 retries)
     h. ncu profile champion (--set full)          → iterv{i}/kernel.ncu-rep
     i. ablate.py: single-method rollback bench    → attribution.json
     j. sass_check.py: verify SASS signatures      → sass_check.json
     k. update state with attribution + SASS results
4. emit summary.md
```

Steps (a), (b), (c), (f), (h), (i), (j) are deterministic — run via scripts.
Steps (d) and (e) are where Claude reasons — follow `references/sm70_fa_catalog.md` and `references/ncu_fa_guide.md`.

---

## Step 0 — Check local environment

Run the env probe **before** doing anything else:

```bash
python <skill>/scripts/check_env.py --out ./env.json
```

It records: GPU name + compute capability (SM arch), nvcc path + version, ncu path + version, CUDA driver, torch + triton versions, GPU peak FLOPS and bandwidth (for roofline), V100-specific SM count and memory config.

If **ncu is not available** or the user lacks `--access=all` perf counters, warn the user — the skill can degrade to benchmark-only mode, but ncu-guided reasoning is significantly weaker.

## Step 0b — Preflight the baseline + ref contract

```bash
python <skill>/scripts/preflight.py \
  --baseline csrc/flash_attn/src/flash_fwd_kernel.h \
  --ref tests/ncu_analyse/test_vllm_flash_attn.py \
  --dims '{"batch":1,"nheads":16,"seqlen":4096,"headdim":128,"causal":false}'
```

Validates:
- FA kernel compiles
- Reference produces correct output (compare with SDPA)
- Benchmark harness works

## Step 1 — Initialize the run folder

```bash
python <skill>/scripts/state.py init \
  --baseline csrc/flash_attn/src/flash_fwd_kernel.h \
  --ref tests/ncu_analyse/test_vllm_flash_attn.py \
  --iterations 3 \
  --ncu-num 5 \
  --branches 4 \
  --dims '{"batch":1,"nheads":16,"seqlen":4096,"headdim":128,"causal":false}' \
  --env ./env.json
```

Creates `./run_YYYYMMDD_HHMMSS/` next to the baseline file with `state.json`:

```jsonc
{
  "run_dir": "...",
  "baseline_file": "...",
  "ref_file": "...",
  "best_file": "<baseline>",
  "best_metric_ms": null,
  "best_ncu_rep": null,
  "env": {...},
  "iterations_total": 3,
  "ncu_num": 5,
  "branches": 4,
  "selected_methods": [],
  "effective_methods": [],
  "ineffective_methods": [],
  "implementation_failed_methods": [],
  "dims": {...},
  "history": [],
  "roofline_history": [],
  "frontier": []
}
```

## Step 2 — Seed `best` with a baseline benchmark

```bash
python <skill>/scripts/run_iteration.py seed-baseline \
  --state ./run_*/state.json
```

## Step 3 — Iteration loop (repeat for i = 1..N)

### 3a. Profile the current `best` with ncu (FULL report — mandatory)

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which best_input
```

Writes `iterv{i}/best_input.ncu-rep` (full ncu report) and `iterv{i}/ncu_top.json`.

**ncu invocation**: Matches `profile.sh` pattern:
```bash
ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:'.*fwd.*' \
    -o "iterv{i}/best_input" \
    python test_vllm_flash_attn.py --flops --use_sdpa
```

### 3b. Compute roofline gaps and axis budgets

```bash
python <skill>/scripts/roofline.py \
  --state ./run_*/state.json \
  --iter $i
```

Reads `ncu_top.json` + `env.json`, computes:
- `Δ_c` = compute utilization gap (1.0 - SM Throughput %)
- `Δ_m` = memory bandwidth utilization gap (1.0 - Memory Throughput %)
- `Δ_l` = max stall percentage (from Warp State Statistics)
- FA-specific: also compute `bank_conflict_severity` and `smem_pressure`

Writes `iterv{i}/roofline.json`:
```jsonc
{
  "delta_compute": 0.85,
  "delta_memory": 0.60,
  "delta_latency": 0.55,
  "bound": "compute",
  "near_peak": false,
  "axis_budget": {"compute": 1, "memory": 1, "latency": 1},
  "bank_conflict_load_pct": 53.1,
  "bank_conflict_store_pct": 17.5,
  "occupancy_pct": 12.48,
  "regs_per_thread": 231,
  "smem_per_block_kb": 98.3
}
```

**Budget allocation rule**: proportional to Δ, rounded, cap per axis = 2, total = 3. If all Δ < 0.15 → `near_peak: true` → **early stop**.

### 3c. Select methods (Claude reasons here)

**Read** (in this order):
1. `references/sm70_fa_catalog.md` — FA-specific optimization catalog for SM70
2. `iterv{i}/roofline.json` — axis budgets, bank conflict data, occupancy
3. `iterv{i}/ncu_top.json` — current bottleneck metrics
4. `state.json` — `best_file`, `selected_methods`, `effective_methods`, `ineffective_methods`
5. The current `best_file` source code (especially `kernel_traits.h` for tile/layout params)
6. `references/ncu_fa_guide.md` — FA-specific metric → root cause mapping

**Selection rule — BUDGET-AWARE PRIORITY SCAN (FA-specific)**:

The FA optimizer uses **trait-level** optimizations. Methods often change compile-time constants in `kernel_traits.h`:

- **SM70_GS_TILE** category: tile sizes (kBlockM, kBlockN, kBlockKSmem, kCtaWarps)
- **SM70_GS_LAYOUT** category: swizzle mode, smem layout atoms, vectorization width
- **SM70_GS_REGISTER** category: register pressure reduction, warp count tuning
- **SM70_GS_PIPELINE** category: sync point optimization, async pattern emulation
- **SM70_GS_TC** category: MMA atom selection, fragment conversion optimization

For each axis with `b_axis > 0`, scan the catalog **from P1 downward**. For each priority level, check:
1. Is `method.id` already in `selected_methods`? → skip (already tried)
2. Does SM70 support the method? → skip if requires sm_80+ feature (cp.async, ldmatrix, etc.)
3. Does the method's **skip condition** apply? → skip (record reason in analysis.md)
4. Does the method's **trigger condition** match the ncu evidence? → skip if no bottleneck here

Select the **first method that passes all four checks**. If `b_axis >= 2`, rank by **trigger strength** and take the top `b_axis`.

**Produce** exactly **B methods** (sum of axis budgets, typically 3). For each, write Chain-of-Thought.

**FA-specific hard constraints**:
1. Tile size changes must be compatible with MMA atom (kWarpRows must be multiple of 8 for `convert_layout_C_to_A_v2`)
2. Shared memory usage must not exceed 96 KB on SM70
3. Register count × threads ≤ 65536 per SM for occupancy
4. `kCtaWarps` change must be reflected in `kNThreads = kCtaWarps * 32`
5. Methods in `ineffective_methods` are **blocked** unless ncu bottleneck fundamentally changed

Save to `iterv{i}/analysis.md` using the template in `templates/iteration_report.md`.

### 3d. Generate K branch kernels (Claude writes code)

All K branches share the **same method combination** from step 3c. They differ in **hyperparameters**:

FA-specific hyperparameters:
- Tile sizes: `kBlockM` (64/128/256), `kBlockN` (32/64/128), `kBlockKSmem` (32/64)
- CTA warps: `kCtaWarps` (2/4/8), affects `kNThreads` and TiledMma layout
- Swizzle: `kSwizzle` (2/3), affects bank conflict patterns
- MMA layout warps: `kMmaLayoutWarps` (row vs column warp distribution)

Write K kernels under `iterv{i}/branches/b{1..K}/`. Each branch modifies only `kernel_traits.h` template parameters. The full kernel file is copied from the best with the trait changes applied.

### 3e. Branch explore: compile + benchmark all K

```bash
python <skill>/scripts/branch_explore.py \
  --state ./run_*/state.json \
  --iter $i
```

Compiles and benchmarks all K branches. Selects champion = fastest valid branch.

**FA-specific**: uses `build.sh` for compilation, reuses the same Python benchmark harness.

### 3f. Repair on validation failure (up to 3 retries per iteration)

If champion fails correctness (vs SDPA reference), Claude rewrites and re-runs 3e. Common FA failures:
- Bank conflict "fix" breaks swizzle → produces wrong results
- Fragment conversion broken by wrong kWarpRows/kBlockN combination
- Shared memory overflow (> 96 KB) → silent corruption

### 3g. Profile champion with ncu (FULL report — mandatory)

```bash
python <skill>/scripts/profile_ncu.py \
  --state ./run_*/state.json \
  --iter $i \
  --which kernel
```

Writes `iterv{i}/kernel.ncu-rep` — **every iteration must have a full ncu report**.

### 3h. Ablation attribution

```bash
python <skill>/scripts/ablate.py \
  --state ./run_*/state.json \
  --iter $i
```

For each method, generates an ablated kernel (champion minus that one method), benchmarks it:
```
attribution(m) = ms_without_m - ms_champion
```

### 3i. SASS verification

```bash
python <skill>/scripts/sass_check.py \
  --state ./run_*/state.json \
  --iter $i
```

Runs `cuobjdump --dump-sass` and checks for expected SASS patterns from `references/sass_fa_signatures.json`.

FA-specific checks:
- HMMA.884.F32.F16 → Tensor Core MMA active
- LDG.E.128.STRONG.GPU → 128-bit vectorized global loads
- STS.128 → 128-bit shared stores (verify swizzle)
- LDS.128 → 128-bit shared loads
- SHFL → warp shuffle for C→A conversion
- MUFU.EX2 → softmax exp2 path

### 3j. Update global state

```bash
python <skill>/scripts/state.py update \
  --state ./run_*/state.json \
  --iter $i \
  --kernel iterv{i}/kernel_traits.h \
  --bench iterv{i}/bench.json \
  --methods-json iterv{i}/methods.json \
  --attribution iterv{i}/attribution.json \
  --sass-check iterv{i}/sass_check.json
```

Rules:
- `selected_methods += all methods` (always)
- Method enters `effective_methods` **only if**: attribution > noise_threshold **AND** SASS verified
- Method enters `implementation_failed_methods` if: SASS check says signature missing
- Method enters `ineffective_methods` if: attribution ≤ noise_threshold but SASS was fine
- If `new_ms < best_ms` by more than noise_threshold → `best_file` updated

---

## Step 4 — Final summary

```bash
python <skill>/scripts/summarize.py \
  --state ./run_*/state.json \
  --out ./run_*/summary.md
```

---

## Reasoning references

- **`references/sm70_fa_catalog.md`** — SM70 FA-specific optimization methods by axis
- **`references/ncu_fa_guide.md`** — How to read ncu output for FA kernels
- **`references/sass_fa_signatures.json`** — Expected SASS patterns for FA methods on SM70
- **`references/kernel_traits_reference.md`** — Complete reference of all `kernel_traits.h` template parameters and their valid ranges

---

## FA-specific failure modes

- **Bank conflict fix produces wrong swizzle** → swizzle changes must maintain data layout semantics; verify with correctness test
- **Tile size breaks MMA layout** → kWarpRows must stay as 8/16/32/64; kBlockN must be 32/64/128/256; assert checks in `convert_layout_C_to_A_v2`
- **Register reduction causes correctness issues** → fewer regs may cause nvcc to spill critical accumulators; verify results
- **Shared memory overflow** → smem > 96 KB causes `cudaErrorInvalidConfiguration`; check `kSmemSize` compile-time assert
- **Benchmark crashes** → check `bench.json` `"error"` field; common due to CUDA OOM or invalid launch config
- **ncu reports all-zero metrics** → permissions issue; run as root or with `--access=all`
- **Build timeouts** → building CUTLASS templates is slow; NEVER kill the build, wait for it

---

## Output contract

```
<project-root>/run_YYYYMMDD_HHMMSS/
├── env.json
├── state.json
├── baseline/
│   ├── flash_fwd_kernel.h    (copied baseline)
│   ├── kernel_traits.h       (copied baseline)
│   └── bench.json
├── iterv1/
│   ├── kernel_traits.h        (champion)
│   ├── analysis.md            (roofline + methods + CoT)
│   ├── methods.json
│   ├── roofline.json
│   ├── best_input.ncu-rep
│   ├── ncu_top.json
│   ├── kernel.ncu-rep
│   ├── attribution.json
│   ├── sass_check.json
│   ├── bench.json
│   └── branches/
│       ├── b1/kernel_traits.h
│       ├── b2/kernel_traits.h
│       ├── b3/kernel_traits.h
│       └── b4/kernel_traits.h
├── iterv2/...
├── iterv3/...
└── summary.md
```

---

## Quick-start for manual analysis

When a user provides an ncu profile output and asks for analysis, use this diagnostic flow:

### SM70 FA Diagnostic Decision Tree

```
1. Check occupancy → < 25% → register or smem pressure
   ├── Register pressure (≥200 regs/thread)
   │   ├── Try: reduce kWarpRows (→ smaller C-fragment → fewer regs)
   │   ├── Try: reduce kCtaWarps (fewer threads → more regs per but fewer total)
   │   └── Try: float→half accumulator precision where safe
   └── Smem pressure (≥90 KB)
       ├── Try: reduce kBlockKSmem (32 vs 64)
       ├── Try: reduce kBlockM (64 vs 128)
       └── Try: Share Q/K smem (if not already)

2. Check bank conflict % → > 30% → swizzle or layout issue
   ├── Load bank conflicts high
   │   ├── Try: change kSwizzle (2→3 or 3→2)
   │   ├── Try: change SmemLayoutAtomQ base layout
   │   └── Try: pad smem by 8 elements per row
   └── Store bank conflicts high
       └── Try: different write swizzle pattern

3. Check MIO throttle % → > 30% → shared memory pipe saturated
   ├── Try: reduce shared store/load frequency
   ├── Try: combine smem operations (wider loads)
   └── Try: use register intermediates more aggressively

4. Check SM throughput → < 50% → stalls or low occupancy
   ├── Eligible warps/sched < 1.0 → stalls dominate
   │   ├── long_scoreboard → global load latency → pipeline better
   │   ├── short_scoreboard → smem latency → ILP/unroll
   │   ├── barrier → sync overhead → reduce sync count
   │   └── mio_throttle → smem pipe → reduce smem traffic
   └── Eligible warps/sched ≥ 1.0 but SM util low → occupancy issue (see #1)

5. Check IPC → < 1.0 on matmul → TC not fully utilized
   ├── Verify HMMA instructions present in SASS
   ├── Check if fragment conversion overhead dominates
   └── MMA atom may need reconfiguration
```

### SM70 FA Profile Hotspot Map

| Bottleneck | Primary Axis | Priority Methods |
|------------|-------------|------------------|
| Bank conflicts > 40% | memory | `bank_conflict_swizzle` (P1), `pad_smem_layout` (P2) |
| Occupancy < 15% | compute | `reduce_register_pressure` (P1), `tune_cta_warps` (P2) |
| MIO throttle > 40% | latency | `widen_smem_access` (P1), `reduce_sync_points` (P2) |
| Eligible warps < 0.5 | latency | `async_copy_emulation` (P1), `pipeline_depth` (P2) |
| TC utilization < 20% | compute | `increase_tile_size` (P1), `warp_specialization` (P2) |
| SM util < 40% + fine IPC | compute | `increase_occupancy` (P1), `thread_coarsening` (P2) |
