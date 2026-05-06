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

ccache -M 10G

echo nproc=$(nproc)

export MAX_JOBS=8
export NVCC_THREADS=2

echo time=$(date '+%F_%H-%M-%S')

python setup.py bdist_wheel 2>&1 | \
  sed -E '/^(\s*(gcc|g\+\+)|\[[0-9 ]*%\]|\[[0-9]+\/[0-9]+\]|copying|running |building )/Id' | \
  sed -E 's/.*([1-9][0-9]* bytes spill stores).*/\x1b[31m&\x1b[0m/g'

echo time=$(date '+%F_%H-%M-%S')

ls -lh dist/vllm*linux*.whl && \
  pip install dist/vllm*linux*.whl 2>&1 | \
  sed -E '/^(Requirement already satisfied|Collecting|Downloading|Processing|  Using cached)/Id' 
