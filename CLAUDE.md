# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

Flash Attention 移植到 V100 (SM70) GPU，移除了 SM80 支持。

- **目标架构**: SM70 (V100)
- **包名**: `vllm_flash_attn`

## benchmark

- **tests/ncu_analyse/profile.sh**: 运行 benchmark 脚本，生成 profile 分析结果。

- **build.sh**: 构建脚本，用于在 Linux 上构建项目。


**重要**: 构建非常耗时，绝对不要终止 build 任务。

## 关键文件

- `csrc/flash_attn/src/flash_fwd_kernel.h` - 前向 kernel 主实现
- `csrc/flash_attn/src/kernel_traits.h` - TiledMMA 定义
- `flash_attn/flash_attn_interface.py` - Python 接口
