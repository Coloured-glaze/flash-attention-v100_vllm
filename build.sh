#!/bin/bash

# source .venv/bin/activate

# cd /workspace/WuTeachingAI/flash-attention-v100/

# export CUDA_HOME=/usr/local/cuda-12.8
# export PATH=/usr/local/cuda-12.8/bin:$PATH
# export D_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
# export PATH=/workspace/WuTeachingAI/conda3/bin:$PATH

# export CMAKE_CXX_COMPILER_LAUNCHER=ccache
# export CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# export CC=$(which gcc-11)
# export CXX=$(which g++-11)

# $CXX --version
nvcc --version

ccache -M 100G

echo nproc=$(nproc)

export MAX_JOBS=8
export NVCC_THREADS=2

# /workspace/WuTeachingAI/conda3/bin/pip uninstall vllm-flash-attn

# /workspace/WuTeachingAI/conda3/bin/pip install --no-build-isolation . -v 2>&1 | \
# sed -E 's/.*[1-9][0-9]* bytes spill stores.*/\x1b[31m&\x1b[0m/g'

# /workspace/WuTeachingAI/conda3/bin/python setup.py bdist_wheel --dist-dir=/workspace/WuTeachingAI/ 2>&1 | \
# sed -E 's/.*[1-9][0-9]* bytes spill stores.*/\x1b[31m&\x1b[0m/g'

echo time=$(date '+%F_%H-%M-%S')

python setup.py bdist_wheel 2>&1 | \
sed -E 's/.*[1-9][0-9]* bytes spill stores.*/\x1b[31m&\x1b[0m/g'

# ls flash-attention-v100/build/temp.linux-x86_64-cpython-312/CMakeFiles/_vllm_fa2_C.dir/csrc/flash_attn/src |grep -n ".cu.o"

echo time=$(date '+%F_%H-%M-%S')

ls dist/vllm*linux*.whl -lh && pip install dist/vllm*linux*.whl 
