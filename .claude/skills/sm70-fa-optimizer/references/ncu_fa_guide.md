# NCU Metrics Guide — Flash Attention Forward Kernel (SM70)

> **Rule of thumb**: SM70 FA kernels are almost always **shared-memory-pipe-bound** or **register-pressure-bound**, not purely compute-bound or DRAM-bound.
>
> This guide maps ncu metrics to FA-specific root causes and remediation methods from `sm70_fa_catalog.md`.

---

## Quick Diagnostic: Read These First

When you see an ncu profile of the FA forward kernel, scan these 5 key metrics first:

### 1. Occupancy & Resource Limits

| Metric | Where to find | Good | Bad → try |
|--------|--------------|------|-----------|
| `launch__registers_per_thread` | Launch Statistics | < 128 | > 200 → `compute.reduce_register_pressure` (P1) |
| `sm__warps_active.avg.pct_of_peak` | Occupancy | > 25% | < 15% → register or smem pressure |
| `launch__occupancy_limit_registers` | Occupancy | if "NO" | if block limit from regs → P1 compute |
| `launch__occupancy_limit_shared_mem` | Occupancy | if "NO" | if block limit from smem → `memory.reduce_smem_usage` (P5) |
| `achieved_occupancy` | Occupancy | > 20% | < 15% + regs high → P1 compute |

**FA interpretation**: 
- 231 regs/thread → 2 blocks/SM max → 12.5% occupancy → **maximum register pressure**
- Each additional block/SM doubles compute throughput opportunity
- Target: 128 regs → 3-4 blocks/SM → 18-25% occupancy
- Target: 96 regs → 5 blocks/SM → 31% occupancy

### 2. Bank Conflicts (FA-specific critical metric)

| Metric | Where to find | Good | Bad → try |
|--------|--------------|------|-----------|
| Shared load bank conflicts (count / wavefronts) | MemoryWorkloadAnalysis_Tables | < 5% | > 20% → `memory.bank_conflict_swizzle` (P1) |
| Shared store bank conflicts (count / wavefronts) | MemoryWorkloadAnalysis_Tables | < 5% | > 10% → `memory.pad_smem_layout` (P2) |
| `l1tex__data_bank_conflicts_shared_load` | Memory Workload Analysis | 0 | > 0 → check swizzle |
| `l1tex__data_bank_conflicts_shared_store` | Memory Workload Analysis | 0 | > 0 → check write layout |

**FA interpretation**:
- 53% load bank conflicts = **every 2nd shared load conflicts** → primary bottleneck
- Source: `SmemLayoutAtomQ` with swizzle mode 3 on kBlockKSmem=64
- 17.5% store bank conflicts = write pattern issue on P → O epilogue
- **Bank conflict formula for FA**: 
  - FP16 = 2 bytes, SM70 bank = 4 bytes → 2 FP16 per bank
  - Stride `Stride<Int<kBlockKSmem>, _1>` on shape `<_8, Int<kBlockKSmem>>` means columns are contiguous
  - With kBlockKSmem=64, thread 0 writes to banks [0,1,2,3,...], thread 1 writes to banks [64,65,66,67,...] ≡ [0,1,2,3,...] → every 8th access aliases

### 3. Stall Reasons (MIO throttle = FA's #1 stall)

| Metric | Where to find | Good | Bad → try |
|--------|--------------|------|-----------|
| `smsp__warp_issue_stalled_mio_throttle.pct` | Warp State Statistics | < 10% | > 30% → `latency.mio_throttle_reduce` (P1) |
| `smsp__warp_issue_stalled_long_scoreboard.pct` | Warp State Statistics | < 10% | > 20% → `latency.pipeline_depth` (P3) |
| `smsp__warp_issue_stalled_short_scoreboard.pct` | Warp State Statistics | < 10% | > 15% → `compute.loop_unroll_pragma` (P6) |
| `smsp__warp_issue_stalled_barrier.pct` | Warp State Statistics | < 5% | > 10% → `latency.reduce_sync_barrier` (P2) |
| `smsp__warp_issue_stalled_wait.pct` | Warp State Statistics | < 5% | > 10% → ILP or loop unroll |

**FA interpretation**:
- **MIO throttle at 46%** = shared memory instruction queue is the bottleneck
  - Every `STS.128` and `LDS.128` goes through MIO pipe
  - 231 registers → 2 warps active → 0.5 eligible → pipe starved but MIO still saturated
  - Fix: reduce shared memory instruction count first (widen, batch, register-ify)
- **long_scoreboard**: global load latency — K/V loads—prefetch helps but limited by sync nature
- **barrier**: `__syncthreads()` overhead between load and compute phases

### 4. Memory Hierarchy

| Metric | Where to find | Good | Bad → try |
|--------|--------------|------|-----------|
| `l1tex__t_sector_hit_rate.pct` | Memory Workload Analysis | N/A for FA | 0.54% is **normal** for streaming FA |
| `lts__t_sector_hit_rate.pct` (L2 hit) | Memory Workload Analysis | > 70% | < 30% → `memory.tiling_smem` (P3) |
| `dram__throughput.pct_of_peak` | Memory Workload Analysis | < 50% | > 90% → `memory.tiling_smem` (P3) |
| `dram__bytes_read.sum` vs expected | Memory Workload Analysis | near (Q+K+V) × blocks | ≫ expected → inefficiency |

**FA interpretation**:
- L1 hit 0.54% is **expected**: FA streams Q/K/V through L1 with `.cg` cache hint
- L2 hit 97.41% is **excellent**: reloading K/V from L2, not DRAM
- DRAM throughput 1.15% → **not DRAM-bound** at all
- This means: DRAM is not the bottleneck; shared memory and compute are

### 5. Compute & Instructions

| Metric | Where to find | Good | Bad → try |
|--------|--------------|------|-----------|
| `sm__pipe_tensor_op_hmma_cycles_active.pct_of_peak` | SM Throughput | > 30% | < 15% → `compute.increase_tile_size` (P2) |
| `sm__inst_executed.avg.per_cycle_active` (IPC) | Compute Workload | > 2.0 | < 1.0 → stalls dominate |
| `smsp__inst_executed_pipe_tensor.sum` | Instruction Stats | high | low → TC not engaged |
| `smsp__inst_executed_pipe_fp32.sum` | Instruction Stats | moderate | high → too much FP32, possibly softmax |
| `smsp__inst_executed_pipe_xu.sum` | Instruction Stats | moderate | high → `compute.softmax_sfu_optimize` (P5) |
| SM Active Cycles | SM Throughput | > 80% | < 50% → occupancy or grid too small |

**FA interpretation**:
- IPC 1.47: still below TC's potential (HMMA can issue every cycle per warp)
- ALU pipeline 27.4% = non-TC work (softmax rescale, fp32→fp16 conversion, address calc)
- TC utilization mixed with SFU and FP32 work → pipeline diversity is natural for FA

---

## FA-Specific Metric Correlations

### When X is high, check Y:

| If this is high... | Check this... | Because... |
|-------------------|---------------|------------|
| Bank conflict loads > 40% | `SmemLayoutAtomQ` swizzle mode | Q/K smem layout determines load pattern for MMA |
| Bank conflict stores > 15% | `SmemLayoutP` layout + epilogue write pattern | P write and O read/write share pattern |
| MIO throttle > 40% | Total STS+LDS instruction count | Shared mem instructions go through MIO pipe |
| MIO throttle + Bank conflict | `kBlockKSmem` choice | kBlockKSmem=64 creates more bank pressure than 32 |
| Regs/thread + Occupancy | `kWarpRows`, `kCtaWarps`, `kBlockN` | Tile sizes directly control register count |
| Occupancy + IPC | Eligible warps per scheduler | Low occupancy → fewer warps to hide latency |
| L2 hit + loop count | `kBlockN` size | Larger kBlockN → fewer iterations → less L2 reload |
| DRAM throughput | `kBlockKSmem` vs `kBlockKGmem` | kBlockKSmem < kBlockKGmem → more smem loads, fewer DRAM loads |

---

## Diagnostic Fast Paths for FA

### Path A: "Occupancy is the root cause" (regs > 200)

```
regs/thread > 200
  → check which is limiting: registers or smem?
    → registers limiting: reduce_register_pressure (P1 compute)
      → reduce kWarpRows (less per-warp work)
      → reduce kCtaWarps (more regs per thread but fewer total)
      → reduce kBlockN (smaller acc_s fragment)
    → smem limiting: reduce_smem_usage (P5 memory)
      → reduce kBlockKSmem
      → reduce kBlockM
```

### Path B: "Bank conflicts are the root cause" (bank conflict > 30%)

```
bank conflict > 30%
  → check which access pattern:
    → load conflicts dominant: bank_conflict_swizzle (P1 memory)
      → change kSwizzle value
      → change SmemLayoutAtomQ base layout
      → add padding to column stride
    → store conflicts dominant: pad_smem_layout (P2 memory)
      → add padding to SmemLayoutP
      → check epilogue write swizzle
```

### Path C: "Stalls are the root cause" (eligible warps < 1.0)

```
eligible warps/sched < 1.0
  → check dominant stall type:
    → MIO throttle > 30%: MIO throttle reduce (P1 latency)
      → reduce shared memory instruction count
      → widen smem access
    → barrier > 15%: reduce_sync_barrier (P2 latency)
      → remove/merge sync points
      → enable Is_Q_in_regs
    → long_scoreboard > 20%: pipeline_depth (P3 latency)
      → double-buffer K/V smem
      → improve prefetch
    → short_scoreboard > 15%: loop_unroll_pragma (P6 compute)
```

### Path D: "Everything looks moderate, but throughput is low"

```
SM util < 60%, IPC > 1.0, bank conflict < 20%, occupancy > 20%
  → "death by a thousand cuts"
  → Try increase_tile_size (P2 compute) to raise computational intensity
  → Try tc_mma_atom_optimize (P3 compute) for better warp distribution
  → Try fragment_conversion_optimize (P4 compute) to reduce shuffle overhead
```

---

## What "Good" Looks Like for SM70 FA

A well-optimized SM70 FA forward kernel should achieve:

| Metric | Current Baseline | Target | Stretch |
|--------|-----------------|--------|---------|
| Occupancy | 12.5% | 20-25% | 30%+ |
| Regs/thread | 231 | ~150 | ~120 |
| Bank conflict (load) | 53.1% | < 20% | < 5% |
| MIO throttle | 46.3% | < 25% | < 15% |
| Eligible warps/sched | 0.50 | 1.0-1.5 | 2.0+ |
| IPC | 1.47 | 2.0+ | 3.0+ |
| SM throughput | 43.4% | 55-65% | 70%+ |
| TC utilization | ? | > 30% | > 50% |
| SM active cycles | ~98% | ~98% | ~98% |

---

## ncu Profile Interpretation Checklist (for FA)

When reading an ncu profile, go through this checklist:

1. [ ] **Read Launch Statistics**: block_size, grid_size, regs/thread, smem/block, waves/SM
2. [ ] **Read Occupancy**: what limits occupancy? Registers or smem?
3. [ ] **Read MemoryWorkloadAnalysis_Tables rules**: any SharedMemoryConflicts?
4. [ ] **Read WarpStateStats rules**: any CPIStall warnings? Which stall type?
5. [ ] **Read SpeedOfLight**: what is the primary bottleneck (compute vs memory)?
6. [ ] **Read SchedulerStats**: issue slot utilization, eligible warps per scheduler
7. [ ] **Read SourceCounters rules**: uncoalesced shared access?
8. [ ] **Read InstructionStats rules**: FP instruction fusion opportunities?
9. [ ] **Compute `delta_compute = 1 - SM_throughput_pct/100`**
10. [ ] **Compute `delta_memory = 1 - Memory_throughput_pct/100`**
11. [ ] **Compute `delta_latency = max_stall_pct/100`**
12. [ ] **Classify bound**: which Δ is largest?
13. [ ] **Allocate budget**: proportional to Δ, cap per axis = 2, total = 3
14. [ ] **Map to catalog**: use diagnostic fast paths above to select methods
