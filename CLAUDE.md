# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Flash Attention 移植到 V100 (SM70) GPU，移除了 SM80 支持。

- **目标架构**: SM70 (V100) 计算能力 7.0 无异步支持
- **包名**: `vllm_flash_attn`

## benchmark

- **profile_run.sh**: 运行 benchmark 脚本，并生成 ncu profile 结果，分析内核瓶颈。

## 构建脚本

- **build.sh**: 构建脚本，用于在 linux shell 上构建项目，执行前需要用户确认。
- 快速编译命令 `export FA_HDIM=128; export CPP_STD=20; export MAX_JOBS=6 && python setup.py bdist_wheel ` 输出内容非常多，需要过滤掉无关内容。

## 其他参考文档

- **.claude/skills/cuda-kernel-optimizer**: CUDA 内核优化技能，包含优化cuda内核的 skills 。
- **cuda_kernel_sample\mlsys2026-flashinfer-contest**: 完全由AI代理驱动的内核优化工作流，包含优化cuda内核和ncu-rep的 skills cuda_kernel_sample\mlsys2026-flashinfer-contest\skills 。
- **cuda_kernel_sample/doc**: 文档目录，包含 v100 参数，其他 v100 flash attention 实现 论文。
- **cuda_kernel_sample/flash-attention-turing_2**: SM75 Turing 版本的 Flash Attention 。
- **cuda_kernel_sample/how-to-optim-algorithm-in-cuda**: 优化 cuda 内核算法的文档，包含cutlass库优化技巧，优化策略。
- **cuda_kernel_sample/references_cuda-12.9_ptx-docs**: cuda 12.9 ptx-isa 文档。
- **cuda_kernel_sample/cuda_LeetCUDA/kernels**: 包含 cuda 内核示例, 包括多种FA实现。
- **cuda_kernel_sample/cuda_learn**: 包含一些cuda内核示例。
- **cuda_kernel_sample/cuda_kernels-community**: 包含huggingface社区收集的 cuda 内核，包含标准的 FA2 实现。
- `csrc\cutlass\include\cute\arch\mma_sm70.hpp`, `csrc\cutlass\include\cute\atom\mma_traits_sm70.hpp`: sm70 mma 布局

**重要**: 构建非常耗时，绝对不要终止 build 任务。

## 关键文件

- `csrc/flash_attn/src/flash_fwd_kernel.h` - 前向 kernel 主实现
- `csrc/flash_attn/src/kernel_traits.h` - TiledMMA 定义
- `csrc/flash_attn/src/utils.h` - 工具函数，tensor布局断言
- `vllm_flash_attn/flash_attn_interface.py` - Python 接口
- `tests/ncu_analyse/test_vllm_flash_attn.py` - 测试用例

## 历史记录
- .workbuddy\memory\*.md - 历史记录文件，包含项目开发历史，包含代码变更，优化策略，。

## 临时文件保存目录
- tmp - 临时文件保存目录，包含编译输出，测试输出等。

## AI 迭代工作流（CUDA 内核优化标准流程）

按以下循环迭代：**定位 → 修改 → 单小算子正确性 → 快速编译 → 安装 → 性能测试 → NCU 排错分析 → 记录历史 → 下一轮**。

### 1. 代码修改（AI 模型改 CUDA 代码）
- 核心改动集中在 `csrc/flash_attn/src/flash_fwd_kernel.h`（kernel 主实现）、`kernel_traits.h`（TiledMMA/tile 配置）、`utils.h`（工具函数、tensor 布局断言）。
- V100 (SM70) 约束：无 cp.async（用 `__ldg`/`SM70_LDG_GLOBAL_CG_128b` 替代）、smem 96KB 限制、无异步支持。改前先查 `cuda_kernel_sample/doc/v100_param.md` 与 `p100_v100_a100_param.md`。
- 参考实现：`cuda_kernel_sample/flash-attention-turing_2`（SM75 同源结构）、`cuda_kernels-community`（标准 FA2）、`cuda_LeetCUDA/kernels`。
- 改动前先读目标代码上下文，明确 fragment 布局与 sync 依赖；不确定的布局变换先写独立小算子验证（见第 6 节）。

### 2. 单小算子测试（先正确性，后性能）
- 测试脚本：`tests/ncu_analyse/test_vllm_flash_attn.py`，与 SDPA/flash_attn 参考实现自动对比 diff。
- 快速验证单点 shape：
  `python tests/ncu_analyse/test_vllm_flash_attn.py --head_dim 128 --batch 1 --n_heads 40 --seq_len 4096 --fa --multi`
- 常用参数：`--causal`、`--kvcache`（含 transposed）、`--seed`、`--multi`（只测一次）。
- **快速编译**（只编译指定 head_dim，大幅缩短时间）：
  `export FA_HDIM=128; export CPP_STD=20; export MAX_JOBS=6 && python setup.py bdist_wheel`
  输出内容非常多，必须过滤：`... 2>&1 | grep -iE "error|warning|spill"`。
  setup.py 还支持 `SHOW_PTX=1`（打印 PTX/寄存器用量）、`NVCC_THREADS`。
- 独立小算子验证模式（仿 `.workbuddy/memory/2026-08-10.md` 的 test_convert_v2.cu）：对可疑的布局/位运算函数单独写 .cu，用 nvcc 编译
  `-std=c++20 -arch=sm_70 -allow-unsupported-compiler -I csrc/cutlass/include -I csrc/flash_attn/src`，与逐元素 reference 做 bit-exact 对比。独立编译需手动补 `#include <optional>` 等。

### 3. whl 编译与信息过滤
- 一键脚本 `build.sh`：sed 过滤只高亮 spill stores 行，编译完自动 `pip uninstall + pip install` 新 whl。
- **重要**：构建非常耗时，绝对不要终止 build 任务；确认后再执行。
- 全量编译（覆盖所有 head_dim）前先用 FA_HDIM 快速验证编译通过，再全量。

### 4. 安装后性能测试
- build.sh 已自动安装；手动方式：`pip uninstall vllm_flash_attn -y && pip install dist/vllm_flash_attn-*-cp31*-linux_x86_64.whl`。
- TFLOPS 对比测试：
  `python tests/ncu_analyse/test_vllm_flash_attn.py --head_dim 128 --flops --fa --sdpa`（加 `--tile` 可对比 tilelang，加 `--plot` 出图）。
- 一键 benchmark + NCU：`./profile_run.sh 128` → 运行 `tests/ncu_analyse/profile.sh`，产出 `prof_时间_commit_dim.log`（含 compact_ncu.py 摘要）与 `profile_out.ncu-rep`。
- 记录 NCU 前后对比表（Duration、spill、DRAM 吞吐、smem bank conflict 等），优化必须以数据为准。

### 5. 查 CUTLASS / PTX 文档
- 项目自带 CUTLASS 源码：`csrc/cutlass/include`（原子布局、copy atom、MMA 定义直接查源码）。
- CUTLASS 优化技巧文档：`cuda_kernel_sample/how-to-optim-algorithm-in-cuda/cutlass`。
- PTX ISA 文档：`cuda_kernel_sample/references_cuda-12.9_ptx-docs`。
- 内核优化知识库：`cuda_kernel_sample/mlsys2026-flashinfer-contest/skills/KernelWiki` 与 `ncu-report-skill`（NCU 报告解读）。
- 内存访问优化：`cuda_kernel_sample/doc/CUDA编程系列(6)：内存访问优化`。

### 6. 排错
- 编译错误：grep 过滤 `error:` 定位首个错误；寄存器/spill 异常用 `SHOW_PTX=1` 重新编译查看。
- 运行时错误：`export CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1` 后复现（profile.sh 已内置），再用
  `$CUDA_HOME/compute-sanitizer/compute-sanitizer --print-limit 1 python tests/ncu_analyse/test_vllm_flash_attn.py --head_dim 128 --fa --multi` 定位越界/竞态。
- 数值错误：先用小 shape（seq_len 128~1024）+ 默认 seed 缩小范围，diff max/mean 定位是 masking、softmax 还是布局转换问题。
- NCU 分析：`ncu --import profile_out.ncu-rep --csv | head -n 200` 再跑 `compact_ncu.py`；瓶颈判断参考 ncu-report-skill。
- 布局断言失败：查 `utils.h` 的 tensor 布局断言，确认 fragment 形状与预期一致。

### 7. 记录历史（每轮迭代必写）
- 写入 `.workbuddy/memory/YYYY-MM-DD.md`，格式参照现有日志：
  - **目标 / 方法 / 环境 / 结果 / 结论 / 编译要点 / 涉及文件（含行号）**
  - NCU 前后对比表（指标 | 实现前 | 实现后 | 变化）
  - **负面结果也要记录并回滚**（如 2026-08-10 的 B1 接入变慢案例），避免重复踩坑。
- 编译要点（flags、include、坑）单独列出，供后续复用。

### 8. 迭代纪律
- 一次只改一个优化点，先正确性再性能；负面结果立即回滚并记录。
- 每轮优化必须带 NCU 数据支撑，无数据不下结论。
- 保留有价值的回归测试文件（如独立验证 .cu）在 tmp 或 tests 目录。