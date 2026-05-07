# SM70 FA Optimizer — Scripts

This skill reuses the parent `cuda-kernel-optimizer` scripts with FA-specific overrides.
The scripts in this directory are thin wrappers that customize:

1. **Build**: uses `build.sh` instead of `nvcc` directly
2. **Benchmark**: uses `tests/ncu_analyse/test_vllm_flash_attn.py` as benchmark harness
3. **ncu profile**: uses kernel name regex `.*fwd.*` from `profile.sh`
4. **Reference**: validated against SDPA (via `--use_sdpa` flag)

## Script Usage

### check_env.py

Probes GPU, nvcc, ncu, CUDA driver.
Additional FA checks: verifies `vllm_flash_attn` can be imported.

```bash
python check_env.py --out ./env.json
```

### preflight.py

Validates baseline kernel compiles and produces correct results vs reference.

```bash
python preflight.py \
  --baseline csrc/flash_attn/src/flash_fwd_kernel.h \
  --ref tests/ncu_analyse/test_vllm_flash_attn.py \
  --dims '{"batch":1,"nheads":16,"seqlen":4096,"headdim":128,"causal":false}'
```

### profile_ncu.py

Runs ncu profile with FA-specific kernel name regex.

```bash
python profile_ncu.py --state state.json --iter 1 --which kernel
```

### roofline.py

Computes roofline gaps with FA-specific metrics (bank conflicts, smem pressure).

### branch_explore.py

Compiles K branch kernels via `build.sh` and benchmarks via `test_vllm_flash_attn.py`.

### ablate.py

Generates ablated kernels by reverting individual methods.

### sass_check.py

Verifies SASS using `cuobjdump --dump-sass` against `references/sass_fa_signatures.json`.

### state.py

Manages the optimization loop state (`state.json`).

### summarize.py

Generates final summary markdown.

## FA-Specific Override Details

### Build override
```
# Instead of: nvcc -arch=sm_70 kernel.cu -o kernel
# Use:
bash build.sh
```
The build.sh script compiles the entire vllm_flash_attn package with the current kernel_traits.h.

### Benchmark override
```
python test_vllm_flash_attn.py --flops --use_sdpa
```
Extracts kernel timing from torch profiler output. Validates correctness against SDPA.

### ncu override
```
ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:'.*fwd.*' \
    -o "profile_out" \
    python test_vllm_flash_attn.py --flops --use_sdpa
```

## Script Dependency Graph

```
orchestrate.py
├── check_env.py → env.json
├── preflight.py (validates baseline+ref)
├── state.py init → state.json + run folder
├── [loop]
│   ├── profile_ncu.py → ncu-rep + ncu_top.json
│   ├── roofline.py → roofline.json (reads ncu_top.json + env.json)
│   ├── [Claude writes branch kernels]
│   ├── branch_explore.py → bench.json (compiles+benchmarks K branches)
│   ├── profile_ncu.py → kernel.ncu-rep (champion)
│   ├── ablate.py → attribution.json
│   ├── sass_check.py → sass_check.json
│   └── state.py update
└── summarize.py → summary.md
```
