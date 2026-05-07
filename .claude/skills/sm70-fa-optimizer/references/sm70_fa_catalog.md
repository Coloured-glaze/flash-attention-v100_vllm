# SM70 Flash Attention Forward Kernel — Optimization Catalog

> **核心原则**：本目录面对 V100 (SM70) Flash Attention 前向 kernel，按真实重要性严格递降排列。
> 等价于 `cuda-kernel-optimizer/references/optimization_catalog.md` 中 Attention/Softmax+GEMM (FA3 style) 行的**SM70 专属展开**。
>
> **选择方法时必须从 P1 开始向下扫描**，选第一个同时满足以下条件的方法：
> 1. ncu 指标显示瓶颈存在
> 2. method id 不在 `state.selected_methods` 中
> 3. 方法兼容 SM70（无 cp.async/ldmatrix/TMA 等 SM80+ 特性）
> 4. 方法的"跳过条件"未被满足
>
> **每个方法对应 `kernel_traits.h` 模板参数或 `flash_fwd_kernel.h` 代码修改**。

---

## Selection Decision Tree (FA-specific)

```
对 axis ∈ {compute, memory, latency}:
  for priority = P1, P2, P3, ...:
    method = fa_catalog[axis][priority]
    if method.id ∈ selected_methods:              → 跳过
    if method.min_sm > 70:                        → 跳过 (SM70不支持)
    if method.arch_feature ∉ SM70:                → 跳过 (如 cp.async/TMA/ldmatrix)
    if method.skip_condition 成立:                  → 跳过
    if method.trigger_condition 不匹配 ncu:        → 跳过
    else:                                          → 选中
```

---

## Memory Axis (FA最关键的轴)

FA kernel 几乎总是 memory-bound 或 shared-memory-pipe-bound。Memory 轴方法通常有最直接的收益。

### P1: `memory.bank_conflict_swizzle` — 共享内存 Bank Conflict 消除（Swizzle 调优）

- **典型收益**: 1.3–2.0× on V100 FA (当前 bank conflict 53%)
- **触发条件**: `l1tex__data_bank_conflicts_*` > 20%；或 ncu rule `SharedMemoryConflicts` 触发
- **跳过条件**: bank conflict 已 < 5%
- **SM70 实现**:
  - 修改 `kSwizzle`：当前值为 `kBlockKSmem == 32 ? 2 : 3`。尝试 `1`/`3`/`4` 不同基数
  - 修改 `SmemLayoutAtomQ` 的 stride：`Layout<Shape<_8, Int<kBlockKSmem>>, Stride<Int<kBlockKSmem>, _1>>` 中调整列 stride (+ padding)
  - 修改 `SmemLayoutAtomP`：当前为 `Layout<Shape<_8, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>`，可加 padding (`Int<kBlockN + 8>`)
  - 修改 `SmemLayoutKV` 的 swizzle 模式：尝试不同 `Swizzle<X, 3, 3>` 参数
- **SM70 限制**:
  - 32 个 bank，bank 宽度 4 字节（FP16 = 2 字节，每 bank 存 2 个 FP16）
  - Swizzle 必须保持数据布局正确性 → 修改后必须通过 correctness test
- **验证**: SASS 中 `STS.128` 的 bank 冲突减少；ncu 中 `l1tex__data_bank_conflicts_*` 显著下降

### P2: `memory.pad_smem_layout` — 共享内存 Padding 消除 Bank Conflict

- **典型收益**: 1.1–1.4×
- **触发条件**: bank conflict 仍 > 15% 但 Swizzle 调优已尝试
- **跳过条件**: bank conflict < 10%
- **SM70 实现**:
  - 给 `SmemLayoutAtomQ`/`SmemLayoutKV`/`SmemLayoutO` 的列维度加 padding
  - 例如 `Layout<Shape<_8, Int<kBlockKSmem + kPadding>>, ...>` 其中 `kPadding = 4 or 8`
  - 给 `SmemLayoutP` 加 padding：`Layout<Shape<_8, Int<kBlockN + 8>>, Stride<Int<kBlockN + 8>, _1>>`
  - **代价**: shared memory 用量增加 padding × rows 字节，需确保不超 96 KB
- **验证**: bank conflict 下降但 smem 用量上升；确保 `kSmemSize` 仍 ≤ 96 KB

### P3: `memory.tiling_smem` — 共享内存 Tiling 策略（Block Size 选择）

- **典型收益**: 1.2–1.8×
- **触发条件**: L2 hit rate < 50%（FA 中不常见，通常 L2 hit 很高）或 dram throughput 意外高
- **跳过条件**: 已有高效 tiling（当前 baseline 已做好）
- **SM70 实现**:
  - 增大 `kBlockM`（128→256）：每 block 算更多行 → 减少 global 读取次数 → 提高计算密度
  - 增大 `kBlockN`（64→128）：每 iteration 算更多列 → 减少 K/V 重读次数
  - 代价：寄存器压力增大，可能降低 occupancy
- **SM70 限制**: `kWarpRows = kBlockM / kCtaWarps` 必须是 8 的整数倍且属于 {8,16,32,64}
- **验证**: smem hit rate 提升；loop iteration count 减少

### P4: `memory.vectorized_access` — 向量化访存宽度

- **典型收益**: 1.05–1.2× per doubling
- **触发条件**: MIO throttle > 30%；LSU 利用率低
- **跳过条件**: 已用 128-bit load/store（当前 baseline 已用 `SM70_LDG_GLOBAL_CG_128b`）
- **SM70 实现**:
  - 当前：128-bit global load (`ld.global.cg.v4.u32` → `LDG.E.128.STRONG.GPU`)
  - **SM70 不支持** 256-bit global load → 无法进一步提升单指令宽度
  - 可考虑：提升 smem load/store 宽度（当前 smem copy atom 为 `AutoVectorizingCopyWithAssumedAlignment<128>`）
  - 可考虑：`GmemTiledCopyQKVPaged` 的 copy atom 布局优化
- **SM70 限制**: Volta 最大 global load 128-bit；smem 最大 128-bit（无 ldmatrix/stmatrix）
- **验证**: `ldst_*` 指令数下降；MIO throttle 下降

### P5: `memory.reduce_smem_usage` — 减少共享内存占用以提升 Occupancy

- **典型收益**: 1.1–1.5× (通过提升 occupancy 间接)
- **触发条件**: `launch__occupancy_limit_shared_mem` 为限制因素；smem > 80 KB 且 occupancy < 20%
- **跳过条件**: smem < 60 KB 或 occupancy 由寄存器限制
- **SM70 实现**:
  - 减少 `kBlockKSmem`（64→32）：减少 Q/K/V smem tile 的列维度
  - 减少 `kBlockM`（128→64）：减少 Q smem tile 的行数
  - 减少 `kBlockN`（64→32）：减少 K/V smem tile 的行数
  - 启用 `Share_Q_K_smem = true`：Q 和 K 共享同一块 smem（但当前为 false，且可能影响性能）
  - **代价**: tile 变小 → 需要更多 loop iteration → 更多 sync
- **SM70 限制**: 96 KB 上限；`kSmemQSize + kSmemKVSize ≤ 96 KB`
- **验证**: smem 占用下降；occupancy 上升（需与寄存器压力共同考虑）

### P6: `memory.cache_modifier` — Global Load Cache Hint 优化

- **典型收益**: 1.02–1.08× (subtle)
- **触发条件**: L1 hit rate < 5% (当前 0.54%) 且 L2 hit rate 够高
- **跳过条件**: 已使用 `.cg` hint（当前 baseline 已用）
- **SM70 实现**:
  - 当前：`SM70_LDG_GLOBAL_CG_128b` 使用 `.cg` (cache global, bypass L1)
  - 可尝试：`.ca` (cache at all levels, keep in L1) → 如果有小量数据热点
  - 可尝试：`.cs` (cache streaming, evict after use) → 如果数据只读一次
  - `.cg` 对 FA 通常最优（因为 Q/K/V 都是 streaming read）
- **验证**: L1 hit rate 变化；DRAM read bytes 变化

### P7: `memory.layout_transpose_optimize` — 内存布局转置优化

- **典型收益**: 1.05–1.3×
- **触发条件**: V 矩阵读取后需转置用于第二个 GEMM；`SmemLayoutVtransposed` 的 swizzle 导致额外 bank conflict
- **跳过条件**: V 转置开销 < 5% 总时间
- **SM70 实现**:
  - 使用 `SmemLayoutVtransposedNoSwizzle` 做 register 级转置，避免 swizzle 干扰
  - 优化 `SmemCopyAtomTransposed` 的 copy layout
  - 使用 warp shuffle 直接做 in-register 转置（而非通过 smem）
- **验证**: V 读取期间的 stall 减少；bank conflict 下降

---

## Compute Axis

### P1: `compute.reduce_register_pressure` — 降低寄存器压力（提升 Occupancy）

- **典型收益**: 1.3–2.0× on V100 FA (当前 occupancy 12.5%，231 regs/thread → 理论最高 2 blocks/SM)
- **触发条件**: `launch__occupancy_limit_registers` 为限制因素；regs/thread > 128
- **跳过条件**: occupancy > 40% 或 regs/thread < 96
- **SM70 实现**:
  - **减少 `kWarpRows`**: 当前可能是 8 或 16，更小的 warp tile = 更少寄存器给 C-fragment (acc_o 和 acc_s)
    - `kWarpRows = 8`: acc_o 和 acc_s 各约 8×kHeadDim 和 8×kBlockN 的寄存器
    - `kWarpRows = 16`: 大约 2× 寄存器 → 但每 warp 算更多 → 更少 sync
  - **减少 `kCtaWarps`**: 更少 warp → 更少总线程 → 每位线程更多寄存器
    - 当前 4 warps → 128 threads；若 2 warps → 64 threads，231 regs/thread 时 occupancy = 65536/(64×231) ≈ 4.4 blocks → 5.5%
    - 若 8 warps → 256 threads，231 regs/thread → 65536/(256×231) ≈ 1.1 blocks → 块无提升
  - **`kMmaLayoutWarps` vs `kCtaWarps`**: 当 kMmaLayoutWarps < kCtaWarps，一些 warp 做 MMA 计算，另一些做数据搬运 → warp specialization 雏形
  - **减少 `kBlockN`**: 更小的 K/V tile → 更小的 acc_s fragment
  - **float→half 局部精度**: acc_o 和 acc_s 的某些中间结果可使用 half（但可能导致数值精度问题）
- **SM70 关键**: 231 regs/thread → `65536 / 231 = 283 threads/SM max` → `ceil(283/128) = 2 blocks/SM` → 12.5% occupancy
- **目标**: 降至 ~128 regs/thread → `65536/128 = 512 threads/SM` → 4 blocks → 25% occupancy
- **目标**: 降至 ~96 regs/thread → `65536/96 = 682 threads/SM` → 5 blocks → 31%
- **验证**: `launch__registers_per_thread` 下降；`sm__warps_active.pct` 上升；IPC 可能因更多 wave 而上升

### P2: `compute.increase_tile_size` — 增大计算 Tile 以提高计算密度

- **典型收益**: 1.1–1.5× (如果当前 tile 太小，如 kBlockM=64)
- **触发条件**: IPC < 1.5 且 SM util < 50% 且 occupancy 足够
- **跳过条件**: occupancy < 15%（tile 增大会增加寄存器压力）
- **SM70 实现**:
  - 增大 `kBlockM`（64→128→256）：更多行同时计算
  - 增大 `kBlockN`（32→64→128）：更大 KV tile
  - 增大 `kBlockKSmem`（32→64）：更少 global load（当前 d=128 时 64 比 128 快 6-10%，因为 bank conflict 更低）
  - 增大 `kCtaWarps`（2→4→8）：更多 warp 分摊计算
- **SM70 限制**: `kWarpRows ∈ {8,16,32,64}`; `kBlockN ∈ {32,64,128,256}`; `kSmemSize ≤ 96 KB`
- **验证**: IPC 上升；loop count 下降

### P3: `compute.tc_mma_atom_optimize` — Tensor Core MMA Atom 选择优化

- **典型收益**: 1.0–1.1× (subtle，当前已使用最优)
- **触发条件**: 使用了非最优 MMA atom
- **跳过条件**: 已使用 `SM70_8x8x4_F32F16F16F32_TN`（SM70 唯一最优选择）
- **SM70 实现**:
  - SM70 的 HMMA 变体：
    - `SM70_8x8x4_F32F16F16F32_TN` (A=RowMajor, B=ColMajor) — **当前使用**，最优
    - `SM70_8x8x4_F32F16F16F32_NT` (A=ColMajor, B=RowMajor) — 需要不同的数据布局
  - TN vs NT 取决于数据在共享内存中的布局
  - `TiledMma` 的 thread layout 影响 warp 之间的工作分配
    - 当前: `Layout<Shape<_1, Int<kMmaLayoutWarps>, _1>>` — 所有 warp 排成行
    - 尝试: `Layout<Shape<Int<kMmaLayoutWarps>, _1, _1>>` — 所有 warp 排成列（如 kBlockN=64 且 4 warps 时分摊 K 维度）
    - 尝试: `Layout<Shape<_2, _2, _1>>` — 2×2 warp tile
- **验证**: SASS 中 HMMA.884 覆盖率；TC utilization 变化

### P4: `compute.fragment_conversion_optimize` — Fragment 布局转换优化（C→A register shuffle）

- **典型收益**: 1.1–1.3×
- **触发条件**: `convert_layout_C_to_A_v2` 中的 warp shuffle 成为热点；大量 `SHFL` 指令
- **跳过条件**: shuffle 开销 < 10%
- **SM70 实现**:
  - 当前 `convert_layout_C_to_A_v2` 使用复杂的 lane_id 计算和 `__shfl_sync` 做 register 间数据重排
  - 优化 lane mapping 函数以减少 shuffle 次数
  - 可尝试通过 shared memory 中转（用 STS+LDS 替代 SHFL，可能在当前 53% bank conflict 场景下不是好主意）
  - 检查 `kMmaThreads == 32` 约束是否可突破（当前限制单 warp MMA 组，因为跨 warp 的 fragment 转换更复杂）
  - 如果 `kWarpRows == 8`，C→A 转换更简单（8行→8×8 MMA 的 A fragment = 8×4）
- **验证**: SHFL 指令数减少；MIO throttle 下降

### P5: `compute.softmax_sfu_optimize` — Softmax SFU 指令优化

- **典型收益**: 1.02–1.08×
- **触发条件**: `smsp__inst_executed_pipe_xu.sum` 高（SFU 利用率高）
- **跳过条件**: 已使用 `exp2` 替代 `exp`
- **SM70 实现**:
  - 用 `exp2f(x * log2(e))` 替代 `expf(x)` → 映射到 `MUFU.EX2` SFU 指令
  - 检查 softmax 实现中 scale 因子的预计算（pow2 格式）
  - 当前 `softmax_rescale_o` 中的实现可能已优化
- **验证**: SFU 指令数变化；SASS 中出现 `MUFU.EX2`

### P6: `compute.loop_unroll_pragma` — 循环展开调优

- **典型收益**: 1.02–1.1×
- **触发条件**: short_scoreboard stall > 10%
- **跳过条件**: 已充分展开（`#pragma unroll` 已大量使用）
- **SM70 实现**:
  - 检查 flash_fwd_kernel.h 中 `#pragma unroll` 的使用
  - 主循环 `for (; n_block >= n_block_min; --n_block)` 展开 —— 可能通过 `#pragma unroll 2` 或完全展开（如果 n_blocks 固定）
  - `gemm` 函数内的循环展开
  - `softmax_rescale_o` 内的循环展开
- **SM70 限制**: 过度展开 → 寄存器溢出 → 反而变慢
- **验证**: short_scoreboard stall 下降；寄存器使用量监控

---

## Latency Axis

### P1: `latency.mio_throttle_reduce` — MIO 管道压力降低

- **典型收益**: 1.2–1.8× (当前 MIO throttle = 46.3% → 最大瓶颈)
- **触发条件**: `smsp__warp_issue_stalled_mio_throttle.pct` > 20%
- **跳过条件**: MIO throttle < 10%
- **SM70 实现**:
  - MIO 管道处理：共享内存指令、特殊数学指令、动态分支
  - **减少共享内存访问频率**:
    - RMS (Register-Memory-Shared) 优化：将数据尽量保持在寄存器中
    - 合并共享内存读/写：一次读更宽的宽度（128-bit）
  - **减少动态分支**:
    - 将 `if (n_block > n_block_min)` 等检查提到循环外或用模板参数消除
    - 将 `mask.template apply_mask<Is_causal, ...>` 的条件分派用 if constexpr
  - **减少 `__syncthreads()` 次数**:
    - 合并连续的 sync 点
    - 使用 warp-level sync 替代 block-level sync 当可能时
- **验证**: MIO throttle 下降；eligible warps 上升

### P2: `latency.reduce_sync_barrier` — 减少 __syncthreads() 同步点

- **典型收益**: 1.1–1.4×
- **触发条件**: barrier stall > 15%；`__syncthreads()` 在热循环中
- **跳过条件**: 无法安全移除 sync（数据依赖）
- **SM70 实现**:
  - 当前 mainloop 中：
    ```cpp
    __syncthreads();               // sync 1: 等 Q 和 K 都加载完 → 可以做 S = QK^T
    FLASH_NAMESPACE::gemm(...);    // S = QK^T (TC)
    __syncthreads();               // sync 2: 等 V 加载完 → 可以做 O += PV
    if (n_block > n_block_min) { copy_K; }
    ```
  - 优化策略：
    - **消除 sync 1**: 如果 Q 已在寄存器中（`Is_Q_in_regs = true`），不需要等 K 加载进 smem。当前为 false。
    - **消除 sync 2**: 如果 V 的转置在寄存器中做，不需要等 V 写入 smem 后的 sync
    - **合并 sync**: 将 sync 1 和 sync 2 合并为一个（如果 gemm 和 K reload 无数据竞争）
    - **warp-level sync 替代**: 如果每个 warp 只访问自己的 smem 分区，可减少 barrier
  - **FENCE 替代 SYNC**: 使用 `__threadfence_block()` 替代 `__syncthreads()` 当只需确保写入可见时
- **SM70 限制**: 无 mbarrier → 只能用 `__syncthreads()` 和 `__syncwarp()`
- **验证**: barrier stall 下降

### P3: `latency.pipeline_depth` — 软件流水线深度

- **典型收益**: 1.2–1.6×
- **触发条件**: long_scoreboard stall > 20%（global load latency 暴露）
- **跳过条件**: 无 long_scoreboard stall
- **SM70 实现**:
  - **无异步拷贝** (no cp.async) → 只能用同步 load + 软件流水线
  - **Prefetch K**: 在当前 iteration 中提前加载下一个 n_block 的 K（已有基本实现：`if (n_block > n_block_min) { copy_K; }`）
  - **Prefetch V**: 类似地提前加载 V
  - **Double buffering**: 分配两个 K smem tile (sK[0], sK[1])，compute on tile 0 while loading tile 1
    - 代价：额外 50% K smem → 需要减少 kBlockN 或 kBlockKSmem 来补偿
  - **Triple buffering**: K、V、Q 各双缓冲 → 代价巨大
- **SM70 限制**: 同步 load → prefetch 后仍要等 load 完成 → 效果有限
- **验证**: long_scoreboard stall 下降

### P4: `latency.warp_specialization_sm70` — SM70 简版 Warp Specialization

- **典型收益**: 1.0–1.2× (SM70 不支持硬件 producer-consumer 分离)
- **触发条件**: 既有 long_scoreboard stall 又有 MIO throttle；不同 warp 的负载不均衡
- **跳过条件**: kCtaWarps < 4（warp 太少无法 specialization）
- **SM70 实现**:
  - 当 `kMmaLayoutWarps < kCtaWarps`：多余的 warp 做数据搬运
  - 当前 baseline 中 `kMmaLayoutWarps = kCtaWarps`：所有 warp 都参与 MMA
  - 尝试 `kMmaLayoutWarps = 2, kCtaWarps = 4`：2 warps 做 MMA，2 warps 做数据搬运
  - 数据搬运 warp 专门加载 K/V/Q 到 smem，MMA warp 专门做计算
  - **问题**: SM70 无 mbarrier → 只能用 `__syncthreads()` 做同步 → 搬运 warp 和计算 warp 互相等待
- **验证**: long_scoreboard stall 和 MIO throttle 的分摊效果；整体 latency

### P5: `latency.early_exit_optimize` — 提前退出优化

- **典型收益**: 1.0–1.05× (for causal attention with large seqlen)
- **触发条件**: causal attention 场景
- **跳过条件**: non-causal attention
- **SM70 实现**:
  - 当前已区分 masking steps 和 non-masking steps
  - 对于 causal attention，当 `n_block_max <= n_block_min` 时提前退出 → 已实现
  - 可优化：更激进的 `n_block_min` 计算（更早退出不必要的 n_blocks）

---

## SM70 FA Optimization Summary Table

| Priority | Method ID | Axis | Typical Gain | Key ncu Trigger |
|----------|-----------|------|-------------|-----------------|
| P1 | `memory.bank_conflict_swizzle` | memory | 1.3–2.0× | bank conflict > 20% |
| P1 | `compute.reduce_register_pressure` | compute | 1.3–2.0× | regs/thread > 128 |
| P1 | `latency.mio_throttle_reduce` | latency | 1.2–1.8× | MIO throttle > 20% |
| P2 | `memory.pad_smem_layout` | memory | 1.1–1.4× | bank conflict > 15% after P1 |
| P2 | `compute.increase_tile_size` | compute | 1.1–1.5× | IPC < 1.5 + occupancy ok |
| P2 | `latency.reduce_sync_barrier` | latency | 1.1–1.4× | barrier stall > 15% |
| P3 | `memory.tiling_smem` | memory | 1.2–1.8× | L2 hit < 50% or dram high |
| P3 | `compute.tc_mma_atom_optimize` | compute | 1.0–1.1× | TC util suboptimal |
| P3 | `latency.pipeline_depth` | latency | 1.2–1.6× | long_scoreboard > 20% |
| P4 | `memory.vectorized_access` | memory | 1.05–1.2× | MIO throttle (LSU) |
| P4 | `compute.fragment_conversion_optimize` | compute | 1.1–1.3× | SHFL instructions hot |
| P4 | `latency.warp_specialization_sm70` | latency | 1.0–1.2× | load imbalance |
| P5 | `memory.reduce_smem_usage` | memory | 1.1–1.5× | smem limiting occupancy |
| P5 | `compute.softmax_sfu_optimize` | compute | 1.02–1.08× | XU pipe high |
| P5 | `latency.early_exit_optimize` | latency | 1.0–1.05× | causal + tail blocks |
| P6 | `memory.cache_modifier` | memory | 1.02–1.08× | L1 hit < 5% |
| P6 | `compute.loop_unroll_pragma` | compute | 1.02–1.1× | short_scoreboard > 10% |
| P7 | `memory.layout_transpose_optimize` | memory | 1.05–1.3× | V transpose costly |

---

## Combining Rules

1. **Bank conflict + register pressure**: 先解决 register pressure（P1 compute），因为寄存器减少可能改变 smem 访问模式（更多 wave → 更多 bank 冲突）
2. **MIO throttle + register pressure**: 同时解决 → register 减少后 MIO throttle 可能自然下降
3. **Pipeline depth + warp specialization**: SM70 上二者互斥 — 要么做流水线（所有 warp 通用），要么做 specialization
4. **Pad smem + reduce smem usage**: 冲突 → pad 增加 smem 用量，需要权衡
5. **Tile size increase + register pressure**: 冲突 → tile 增大 → 更多寄存器 → 只能选其一
6. **Swizzle 变更**: 任何改变 swizzle 的方法都必须在 correctness test 中验证数据正确性

---

## SM70 不支持的方法（来自通用 catalog）

以下通用优化方法在 SM70 上**不可用**（理由）：

| Method | 原因 |
|--------|------|
| `memory.async_copy` (通用 P5) | SM70 无 `cp.async` 指令 |
| `memory.multi_stage_pipeline` (通用 P6) | 依赖 cp.async + mbarrier |
| `memory.ldmatrix_stmatrix` (通用 P9) | SM70 无 `ldmatrix`/`stmatrix` |
| `memory.tma_*` (通用 P11-12, P20-22) | SM70 无 TMA |
| `memory.epilogue_visitor_tree_fusion` (通用 P13) | SM70 无 TMA epilogue |
| `memory.distributed_smem` (通用 P21) | SM70 无分布式共享内存 |
| `memory.smem_register_spilling` (通用 P19) | 需要 CUDA ≥ 13 |
| `compute.warp_specialization` (通用 P3) | SM70 无 mbarrier / setmaxnreg |
| `compute.gemm_softmax_interleave` (通用 P8) | SM70 无独立 warpgroup 机制 |
| `latency.async_pipeline` (通用 P1) | 依赖 cp.async |
| `latency.asymmetric_mbarrier_sync` (通用 P9) | SM70 无 mbarrier |
| `latency.warp_shuffle_sync` (通用 P3) | SM70 有 `__shfl_sync` 但 FA 已用 |
| `latency.tile_scheduler_swizzle` (通用 P4) | 通用优化，FA 通过 grid launch 间接可用 |
| `latency.pingpong_warpgroup_schedule` (通用 P10) | SM70 无独立 warpgroup |
| `latency.intra_wg_gemm_softmax_pipeline` (通用 P12) | SM70 无此机制 |
