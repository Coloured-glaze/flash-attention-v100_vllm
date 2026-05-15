
time=$(date '+%y%m%d_%H%M')
git_commit=$(git rev-parse --short HEAD)
file_name="prof"

echo "git commit: ${git_commit}" > ${file_name}_${time}_${git_commit}.txt

export CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1

echo "start benchmark at ${time}" \
&& \
compute-sanitizer --print-limit 1 python test_vllm_flash_attn.py --flops --flops_num 5 --fa \
&& \
export CUDA_LAUNCH_BLOCKING=0 TORCH_USE_CUDA_DSA=0 \
&& \
python test_vllm_flash_attn.py --flops --profile --fa --sdpa >> ${file_name}_${time}_${git_commit}.txt \
&& \
echo "benchmark done at ${time}"

echo "start profile analysis"


KERNEL_REGEX='.*fwd.*'

ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:"${KERNEL_REGEX}" \
    -o "profile_out" \
    python test_vllm_flash_attn.py --flops --fa \
&& \
ncu --import profile_out.ncu-rep --csv | head -n 200 > ${file_name}_${time}.csv \
&& \
python compact_ncu.py ${file_name}_${time}.csv >> ${file_name}_${time}_${git_commit}.txt \
&& \
rm ${file_name}_${time}.csv \
&& \
echo "profile analysis done. "
echo "result saved to ${PWD}/${file_name}_${time}_${git_commit}.txt"


