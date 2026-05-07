# kernel_traits.h Template Parameter Reference (SM70 FA Forward)

> Complete reference of all compile-time template parameters in `Flash_fwd_kernel_traits` that can be tuned for SM70 optimization.

---

## Template Parameters

```cpp
template<
    int kHeadDim_,      // head dimension (32, 64, 96, 128, 192, 256)
    int kBlockM_,        // Q tile rows per block (64, 128, 256)
    int kBlockN_,        // K/V tile rows per block (32, 64, 128)
    int kCtaWarps_,      // warps per CTA (2, 4, 8)
    int kMmaLayoutWarps_ // warps in MMA layout (1..kCtaWarps_, default = kCtaWarps_)
>
struct Flash_fwd_kernel_traits { ... };
```

### Default template arguments used in code:
```cpp
Flash_fwd_kernel_traits<128, 64, 64, 4, 4>
// kHeadDim=128, kBlockM=64, kBlockN=64, kCtaWarps=4, kMmaLayoutWarps=4
```

---

## Parameter: `kHeadDim_`

The head dimension of the attention computation.

| Value | Valid | Notes |
|-------|-------|-------|
| 32 | ✅ | Small heads, rarely used |
| 64 | ✅ | Common |
| 96 | ✅ | Less common |
| 128 | ✅ | Most common (default in benchmarks) |
| 192 | ✅ | Large heads |
| 256 | ✅ | Very large heads |

**Constraints**:
- `kHeadDim % 32 == 0` (static_assert)
- `kHeadDim % kGmemElemsPerLoad == 0` (static_assert, 128-bit = 8 FP16)

**Derived values**:
```cpp
kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;  // smem tile width for Q/K
kBlockKGmem = kHeadDim % 128 == 0 ? 128 : (kHeadDim % 64 == 0 ? 64 : 32);  // gmem tile width for Q/K
kSwizzle = kBlockKSmem == 32 ? 2 : 3;  // swizzle base
```

**Trade-offs**:
- Larger → more smem needed, more MMA iterations per tile
- `kBlockKSmem = 64` for d=128: 6-10% faster than `kBlockKGmem=128` despite more gmem loads (better bank conflict pattern)
- `kBlockKSmem = 32` for d=32/64: less smem, potentially faster due to different swizzle (kSwizzle=2 vs 3)

---

## Parameter: `kBlockM_`

Number of Q rows processed per threadblock (block tile dimension in M).

| Value | Valid | kWarpRows | Notes |
|-------|-------|-----------|-------|
| 64 | ✅ | 64/kCtaWarps | Small tile, less smem, more blocks |
| 128 | ✅ | 128/kCtaWarps | Default for d=128 |
| 256 | ✅ | 256/kCtaWarps | Large tile, more registers |

**Constraints**:
- `kBlockM % kCtaWarps == 0`
- `kWarpRows ∈ {8, 16, 32, 64}` (for `convert_layout_C_to_A_v2`)

**Example valid combinations**:
```
kBlockM=64,  kCtaWarps=4 → kWarpRows=16 ✅
kBlockM=64,  kCtaWarps=2 → kWarpRows=32 ✅
kBlockM=64,  kCtaWarps=8 → kWarpRows=8  ✅
kBlockM=128, kCtaWarps=4 → kWarpRows=32 ✅
kBlockM=128, kCtaWarps=8 → kWarpRows=16 ✅
kBlockM=128, kCtaWarps=2 → kWarpRows=64 ✅
kBlockM=256, kCtaWarps=2 → kWarpRows=128 ❌ (not in {8,16,32,64})
kBlockM=256, kCtaWarps=4 → kWarpRows=64 ✅
kBlockM=256, kCtaWarps=8 → kWarpRows=32 ✅
```

**Trade-offs**:
- Larger kBlockM → more rows per block → fewer global Q reads → better arithmetic intensity
- Larger kBlockM → larger acc_o fragment → more registers
- Larger kBlockM → larger Q smem → more smem pressure

---

## Parameter: `kBlockN_`

Number of K/V rows processed per inner loop iteration.

| Value | Valid | Notes |
|-------|-------|-------|
| 32 | ✅ | Small KV tile, more iterations |
| 64 | ✅ | Default, balanced |
| 128 | ✅ | Large KV tile, may increase regs |

**Constraints**:
- `kBlockN ∈ {32, 64, 128, 256}` (for `convert_layout_C_to_A_v2`)
- Larger kBlockN = larger acc_s fragment = more registers

**Trade-offs**:
- Larger → fewer outer loop iterations → less K/V reload
- Larger → larger acc_s → more register pressure → lower occupancy
- Larger → more smem for P tile → more bank conflicts possible

---

## Parameter: `kCtaWarps_`

Number of warps per CTA (threadblock). Determines `kNThreads = kCtaWarps * 32`.

| Value | kNThreads | Registers @231/thread | Blocks/SM max |
|-------|-----------|----------------------|---------------|
| 2 | 64 | 65536/(64×231) = 4.4 → 4 | 4 → 32% occupancy |
| 4 | 128 | 65536/(128×231) = 2.2 → 2 | 2 → 12.5% occupancy |
| 8 | 256 | 65536/(256×231) = 1.1 → 1 | 1 → 6.25% occupancy |

**Constraints**:
- `kNThreads % kGmemThreadsPerRow == 0`
- `kNThreads % kMmaThreads == 0`
- `kMmaThreads % 32 == 0` → TiledMma uses entire warps

**Trade-offs**:
- More warps → more parallelism → better at hiding latency → more eligible warps
- More warps → but with fixed regs/thread, total regs = threads × regs/thread → occupancy limited
- **Key insight**: with 231 regs/thread, `kCtaWarps=4` (128 threads) is at the boundary: 128×231=29568 → 2 blocks/SM → 12.5%
- `kCtaWarps=2` (64 threads): 64×231=14784 → ~4 blocks/SM → 25% occupancy → **higher!**
- But fewer threads = each thread does more work = may need even more registers

---

## Parameter: `kMmaLayoutWarps_`

How warps are distributed in the TiledMMA layout.

```cpp
using TiledMma = TiledMMA<
    MMA_Atom_Arch,  // SM70_8x8x4_F32F16F16F32_TN
    Layout<Shape<_1, Int<kMmaLayoutWarps>, _1>>,  // thread layout
    Tile<Int<kWarpRows>, _16, _4>  // MMA tile per warp
>;
```

- `kMmaLayoutWarps = kCtaWarps` (default): all warps do MMA
- `kMmaLayoutWarps < kCtaWarps`: extra warps could do data movement
  - But on SM70, there's no hardware support for producer-consumer separation

**Trade-offs**:
- Currently not a significant optimization lever on SM70

---

## Derived Constants Reference

### Shared Memory Layout

```cpp
// Q smem: (kBlockM, kHeadDim) swizzled
SmemLayoutQ = tile_to_shape(
    composition(Swizzle<kSwizzle, 3, 3>{},
                Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}),
    Shape<Int<kBlockM>, Int<kHeadDim>>{});

// K/V smem: (kBlockN, kHeadDim) swizzled
SmemLayoutKV = tile_to_shape(
    composition(Swizzle<kSwizzle, 3, 3>{},
                Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>{}),
    Shape<Int<kBlockN>, Int<kHeadDim>>{});

// P smem: (kBlockM, kBlockN) row-major (no swizzle)
SmemLayoutP = tile_to_shape(
    Layout<Shape<_8, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>{},
    Shape<Int<kBlockM>, Int<kBlockN>>{});

// O smem: (kBlockM, kHeadDim) swizzled
SmemLayoutO = ... // similar to SmemLayoutQ
```

### Smem Size Calculation

```cpp
kSmemQSize = kBlockM × kHeadDim × sizeof(half)      // Q tile
kSmemKVSize = kBlockN × kHeadDim × 2 × sizeof(half)  // K + V tiles (each kBlockN × kHeadDim)
kSmemPSize = kBlockM × kBlockN × sizeof(half)        // P tile (reuses Q space for output)
kSmemSize = kSmemQSize + kSmemKVSize                  // ≤ 96 KB

// Example: kBlockM=64, kBlockN=64, kHeadDim=128
// kSmemQSize = 64×128×2 = 16 KB
// kSmemKVSize = 64×128×2×2 = 32 KB
// kSmemSize = 16+32 = 48 KB → well under 96KB ✅
```

### Register Usage Estimator

```
Main register consumers:
  acc_o: kWarpRows × kHeadDim floats     = kWarpRows × 128 × 4 bytes = kWarpRows × 512 bytes → kWarpRows × 128 regs
  acc_s: kWarpRows × kBlockN floats       = kWarpRows × 64 × 4 bytes = kWarpRows × 256 bytes → kWarpRows × 64 regs
  cvt_C_to_A: kWarpRows × kBlockN/32 × 2 halfs = kWarpRows × kBlockN × 2/32 × 2 bytes → small
  gmem_tiled_copy registers: ~16-24 regs
  MMA fragment registers: ~16-32 regs
  misc (thread index, loop counter, etc.): ~8-16 regs

Estimated: ~kWarpRows×(128+64) + 32 + 16 ≈ kWarpRows×192 + 48

kBlockM=64,  kCtaWarps=4,  kWarpRows=16: 16×192+48 ≈ 3120 bits ≈ 97 regs (per thread? No, this is fragment total)
                                             Actually: acc_o = 16×128/32 = 64 regs/thread, acc_s = 16×64/32 = 32 regs/thread
                                             Total: ~64+32+48 ≈ 144 regs/thread

kBlockM=128, kCtaWarps=4,  kWarpRows=32: acc_o = 32×128/32=128 regs, acc_s = 32×64/32=64 regs
                                             Total: ~128+64+48 ≈ 240 regs/thread ← matches observed 231
```

---

## Pareto-Optimal Configurations

For d=128, these configurations have been validated (or are theoretically sound):

| kBlockM | kBlockN | kCtaWarps | kWarpRows | kNThreads | ~Regs | Occupancy | Smem |
|---------|---------|-----------|-----------|-----------|-------|-----------|------|
| 128 | 64 | 4 | 32 | 128 | 231 | 12.5% | ~48KB |
| 64 | 64 | 4 | 16 | 128 | ~150 | 18.75% | ~32KB |
| 64 | 64 | 2 | 32 | 64 | ~200 | 25% | ~32KB |
| 128 | 32 | 4 | 32 | 128 | ~200 | 12.5% | ~40KB |
| 128 | 64 | 2 | 64 | 64 | ~300 | 6.25% | ~48KB |
| 256 | 64 | 4 | 64 | 128 | ~350 | 12.5% | ~64KB |

---

## Key Tuning Levers (Priority-Ordered)

1. **`kWarpRows` (via kBlockM / kCtaWarps)**: #1 register pressure control
   - kWarpRows = 8: ~120 regs → highest occupancy → fastest if occupancy matters
   - kWarpRows = 16: ~150 regs → balanced
   - kWarpRows = 32: ~230 regs → current default → lowest occupancy but highest per-thread work

2. **`kSwizzle` (via kBlockKSmem)**: #1 bank conflict control
   - kSwizzle = 2 (kBlockKSmem=32): simpler pattern, possibly fewer bank conflicts
   - kSwizzle = 3 (kBlockKSmem=64): current, 53% bank conflict

3. **`kBlockN`**: #2 register pressure + loop count trade-off
   - kBlockN = 32: small acc_s, more loops
   - kBlockN = 64: current, balanced
   - kBlockN = 128: larger acc_s, fewer loops, more regs

4. **`kCtaWarps`**: occupancy control
   - kCtaWarps = 2: higher occupancy, more regs per thread (64 threads → 65536/64 ~ 1024 regs available per thread)
   - kCtaWarps = 4: current, balanced
   - kCtaWarps = 8: lower occupancy, more parallelism
