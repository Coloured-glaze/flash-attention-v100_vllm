# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Flash Attention 移植到 V100 (SM70) GPU，移除了 SM80 支持。

- **目标架构**: SM70 (V100)
- **包名**: `vllm_flash_attn`

## benchmark

- **profile_run.sh**: 运行 benchmark 脚本，并生成 profile 结果，分析内核瓶颈。

## 构建脚本

- **build.sh**: 构建脚本，用于在 shell 上构建项目，执行前需要用户确认。

## 其他参考文档

- **.claude/skills/cuda-kernel-optimizer**: CUDA 内核优化技能，包含优化cuda内核的 skills 。
- **.claude/skills/sm70-fa-optimizer**: SM70 Flash Attention 优化技能，包含优化fa内核的 skills 。
- **cuda_kernel_sample/doc**: 文档目录，包含 v100 参数，其他 v100 flash attention 实现 论文。
- **cuda_kernel_sample/flash-attention-turing_2**: SM75 Turing 版本的 Flash Attention 。
- **cuda_kernel_sample/cuda_LeetCUDA/kernels**: 包含大量 cuda 内核示例, 包括多种FA实现。
- **cuda_kernel_sample/cuda_LeetCUDA/ffpa-attn**: 支持 512 DIM FA 实现。
- **cuda_kernel_sample/cuda_learn**: 包含 cuda 内核示例。
- **cuda_kernel_sample/cuda_kernels-community**: 包含hf社区收集的 cuda 内核，包含标准的 FA2 FA3 FA4 实现。。

**重要**: 构建非常耗时，绝对不要终止 build 任务。

## 关键文件

- `csrc/flash_attn/src/flash_fwd_kernel.h` - 前向 kernel 主实现
- `csrc/flash_attn/src/kernel_traits.h` - TiledMMA 定义
- `csrc/flash_attn/src/utils.h` - 工具函数，tensor布局断言
- `vllm_flash_attn/flash_attn_interface.py` - Python 接口
