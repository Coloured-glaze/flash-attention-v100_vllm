
head_dim=${1:-128}
time=$(date '+%y%m%d_%H%M')
git_commit=$(git rev-parse --short HEAD)
output_file=prof_${time}_${git_commit}_${head_dim}.log

echo "git commit: ${git_commit}" > ${output_file}

export CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1

echo "start compute-sanitizer at ${time}" \
&& \
$CUDA_HOME/compute-sanitizer/compute-sanitizer --print-limit 1 python test_vllm_flash_attn.py --head_dim $head_dim --flops --flops_num 5 --fa \
&& \
export CUDA_LAUNCH_BLOCKING=0 TORCH_USE_CUDA_DSA=0 \
&& \
python test_vllm_flash_attn.py --head_dim $head_dim --flops --profile --fa --sdpa >> ${output_file} \
&& \
echo "compute-sanitizer done at ${time}"

echo "start profile analysis"


KERNEL_REGEX='.*fwd.*'

ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:"${KERNEL_REGEX}" \
    -o "profile_out" \
    python test_vllm_flash_attn.py --head_dim $head_dim --flops --fa \
&& \
ncu --import profile_out.ncu-rep --csv | head -n 200 > ${output_file}.csv \
&& \
python compact_ncu.py ${output_file}.csv >> ${output_file} \
&& \
rm ${output_file}.csv \
&& \
echo "profile analysis done. "
echo "result saved to ${PWD}/${output_file}"


