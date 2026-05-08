#!/bin/bash

# source .venv/bin/activate

# cd /workspace/WuTeachingAI/flash-attention-v100/

# export CUDA_HOME=/usr/local/cuda-12.8
# export PATH=/usr/local/cuda-12.8/bin:$PATH
# export D_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
# export PATH=/workspace/WuTeachingAI/conda3/bin:$PATH

# export CC=$(which gcc-11)
# export CXX=$(which g++-11)

# $CXX --version
nvcc --version

# Install ninja and sccache if not present
if ! command -v ninja &> /dev/null; then
    echo "Installing ninja..."
    pip install ninja -q 
fi

if ! command -v sccache &> /dev/null; then
    echo "Installing sccache..."
    pip install sccache -q 
fi

ccache -M 20G

# Configure sccache
sccache --start-server

echo nproc=$(nproc)

free -m

export MAX_JOBS=12
export NVCC_THREADS=2
# export CMAKE_BUILD_TYPE=Release

target_whl=dist/vllm_flash_attn-2.7.2.post1-cp312-cp312-linux_x86_64.whl

rm ${target_whl} 2>&1 || true

time=$(date '+%F_%H:%M:%S')
echo "start build at ${time}"

python setup.py bdist_wheel 2>&1 | \
  sed -E 's/.*([1-9][0-9]* bytes spill stores).*/\x1b[31m&\x1b[0m/g' && \
echo "start build at ${time} -- end build at $(date '+%F_%H-%M-%S')" && \
ls -lh ${target_whl} && \
  pip uninstall vllm_flash_attn -y && pip install ${target_whl} 

sccache --stop-server
