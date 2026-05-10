
if [ -n "$1" ]; then
  export FA_HDIM=$1
  echo "FA_HDIM: ${FA_HDIM}"
fi

nvcc --version

# Install ninja and sccache if not present
if ! command -v ninja &> /dev/null; then
    echo "Installing ninja..."
    pip install ninja -q 
fi

echo nproc=$(nproc)

free -m

export MAX_JOBS=12
export NVCC_THREADS=2
# export CMAKE_BUILD_TYPE=Release

system=$(python -c "import platform; print(platform.system().lower())")

target_whl_linux=dist/vllm_flash_attn-2.7.2.post1-cp31*-cp31*-linux_x86_64.whl
target_whl_win=dist/vllm_flash_attn-2.7.2.post1-cp31*-cp31*-win_amd64.whl

if [ "${system}" == "windows" ]; then
  target_whl=${target_whl_win}
else
  target_whl=${target_whl_linux}

  ccache -M 20G
  if ! command -v sccache &> /dev/null; then
      echo "Installing sccache..."
      pip install sccache -q 
  fi

  sccache --start-server # Configure sccache
fi

rm ${target_whl} 2>&1 || true

time=$(date '+%F_%H:%M:%S')
echo "start build at ${time}"

python setup.py bdist_wheel 2>&1 | \
  sed -E 's/.*([1-9][0-9]* bytes spill stores).*/\x1b[31m&\x1b[0m/g' && \
echo "start build at ${time} -- end build at $(date '+%F_%H-%M-%S')" && \
ls -lh ${target_whl} && \
  pip uninstall vllm_flash_attn -y && pip install ${target_whl} 

if [ "${system}" != "windows" ]; then
  sccache --stop-server
fi