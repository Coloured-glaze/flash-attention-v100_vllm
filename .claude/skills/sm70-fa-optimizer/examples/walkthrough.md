# Example: Analyzing the Current Baseline Profile

This walkthrough uses the real ncu profile from `tests/ncu_analyse/profile_out.txt` to demonstrate how the SM70 FA Optimizer reasons about bottlenecks.

---

## Step 1: Read the Profile

From `profile_out.txt`, the kernel `flash_fwd_kernel<Flash_fwd_kernel_traits<128, 64, 64, 4, 4>, ...>` shows:

### Key Metrics

| Section | Metric | Value |
|---------|--------|-------|
| GPU Speed Of Light | Memory Throughput | 89.77% |
| GPU Speed Of Light | SM Throughput | 43.37% |
| GPU Speed Of Light | Duration | 17.80 ms |
| Occupancy | Theoretical Occupancy | 12.50% |
| Occupancy | Achieved Occupancy | 12.48% |
| Occupancy | Registers Per Thread | 231 |
| Occupancy | Shared Memory | 98.30 KB |
| Launch | Block Size | 128 threads |
| Launch | Grid Size | 2560 blocks |

### Stalls

| Metric | Value |
|--------|-------|
| MIO Throttle | 46.3% of stall cycles |
| Eligible Warps/Sched | 0.50 (out of 2.00 active) |
| Issued Warp/Sched | 0.37 |
| Warp Cycles/Instruction | 5.42 |

### Bank Conflicts

| Type | Rate | Severity |
|------|------|----------|
| Shared Load | 2.1-way (53.10%) | **CRITICAL** |
| Shared Store | 4.6-way (17.54%) | **MODERATE** |

### Memory Hierarchy

| Cache | Hit Rate |
|-------|----------|
| L1/TEX | 0.54% |
| L2 | 97.41% |
| DRAM Throughput | 1.15% |

---

## Step 2: Classify the Bottleneck

### Roofline Analysis

```
Δ_compute = 1 - 0.4337 = 0.5663
Δ_memory  = 1 - 0.8977 = 0.1023
Δ_latency = 0.463 (MIO throttle = max stall)
```

**Bound: Latency (stall-dominated)**

The kernel is:
- **NOT compute-bound** (43% SM util, has headroom)
- **NOT DRAM-bound** (1.15% DRAM throughput — almost all data comes from L2)
- **SHARED MEMORY PIPE BOUND** (46% MIO throttle + 53% bank conflicts)

### Why is SM throughput 43% when memory is 90%?

Memory Throughput here refers to **shared memory bandwidth**, not DRAM. SM70 reports "Memory Throughput" as the combined L1/smem bandwidth utilization. At 90%, the shared memory pipeline is saturated — but because only 0.50 warps are eligible per cycle, the SM's compute units are idle 57% of the time.

### Root Cause Chain

```
231 regs/thread → only 2 blocks/SM → only 2 warps active/sched
→ only 0.50 warps eligible (rest stalled on MIO)
→ shared memory pipe saturated at 90%
→ compute units idle 57% of the time
→ SM throughput only 43%
```

It's a **register-pressure → occupancy → stall → throughput** chain.

---

## Step 3: Budget Allocation

```
B = 3 total
Δ_latency = 0.463  (proportional: 0.463/1.132 = 0.41 → 1)
Δ_compute = 0.566  (proportional: 0.566/1.132 = 0.50 → 1)
Δ_memory  = 0.102  (proportional: 0.102/1.132 = 0.09 → 1, minimum)
But total = 3, Δ_memory is small → cap adjusted to 0
Recalibrated:
  compute = 2 (register pressure is the root cause)
  latency = 1 (MIO throttle is the immediate symptom)
  memory  = 0 (DRAM is not the issue, bank conflicts are addressed under latency)
```

**Budget**: compute=2, latency=1, memory=0

---

## Step 4: Priority Scan

### Compute Axis (budget=2)

1. **P1**: `compute.reduce_register_pressure`
   - Trigger: regs/thread=231 > 128 ✅, occupancy=12.5% < 25% ✅
   - Not in selected_methods ✅
   - SM70 compatible ✅
   - **SELECTED** ← budget slot 1

2. **P2**: `compute.increase_tile_size`
   - Trigger: IPC=1.47 < 1.5 → marginal trigger
   - Occupancy=12.5% < 15% → SKIP (skip condition: occupancy < 15% when increase tile would increase registers)
   - **SKIPPED**: occupancy too low to increase tile without making register pressure worse

3. **P3**: `compute.tc_mma_atom_optimize`
   - Trigger: already using optimal atom SKIP
   - **SKIPPED**: already optimal

4. **P4**: `compute.fragment_conversion_optimize`
   - Trigger: SHFL instructions in hot path
   - **SELECTED** ← budget slot 2 (after P1, P2 skipped, P3 skipped)

### Latency Axis (budget=1)

1. **P1**: `latency.mio_throttle_reduce`
   - Trigger: MIO throttle=46.3% > 20% ✅
   - Not in selected_methods ✅
   - SM70 compatible ✅
   - **SELECTED** ← budget slot 1

### Memory Axis (budget=0)

Memory axis has budget=0 → no methods selected. Bank conflict resolution is expected to come as a side-effect of register pressure reduction (different smem access pattern with different kWarpRows).

---

## Step 5: Method Selection Summary

| # | Method | Axis | Priority | Reasoning |
|---|--------|------|----------|-----------|
| 1 | `compute.reduce_register_pressure` | compute | P1 | 231 regs → root cause of low occupancy |
| 2 | `compute.fragment_conversion_optimize` | compute | P4 | SHFL chain for C→A conversion is 2nd compute bottleneck |
| 3 | `latency.mio_throttle_reduce` | latency | P1 | 46% MIO throttle → shared memory pipe saturated |

---

## Step 6: Implementation Plan

### Method 1: `compute.reduce_register_pressure`
- **Change**: kBlockM=128→64, kWarpRows=32→16 (keep kCtaWarps=4)
- **Why**: kWarpRows=16 halves acc_o (16×128/32=64 regs vs 128) and acc_s (16×64/32=32 vs 64)
- **Expected**: regs ~150, occupancy ~18.75% (3 blocks/SM)
- **Risk**: 2× more m_blocks → 2× more Q loads from gmem → more global load traffic

### Method 2: `compute.fragment_conversion_optimize`
- **Change**: With kWarpRows=16, the C→A conversion uses 2 row groups (kRowGroups=2) instead of 4; SHFL count reduces proportionally
- **Expected**: ~30% fewer SHFL instructions per conversion
- **Risk**: None — conversion function is size-parameterized

### Method 3: `latency.mio_throttle_reduce`
- **Change**: Merge two consecutive `__syncthreads()` in the main loop where safe
- **Expected**: barrier stall reduction, fewer MIO instructions for barrier
- **Risk**: Must verify no data race introduced

---

## Step 7: Branch Generation

| Branch | kBlockM | kBlockN | kCtaWarps | kWarpRows | Strategy |
|--------|---------|---------|-----------|-----------|----------|
| b1 | 64 | 64 | 4 | 16 | Conservative: smaller M tile only |
| b2 | 64 | 64 | 2 | 32 | Aggressive reg reduction: fewer warps → more regs/thread → need to verify |
| b3 | 128 | 32 | 4 | 32 | Smaller N tile: reduce acc_s size |
| b4 | 64 | 128 | 4 | 16 | Larger N: fewer iterations, more regs for acc_s |

Each branch also applies:
- Fragment conversion optimization (automatic from kWarpRows change)
- MIO throttle reduction (sync merge in main loop)
