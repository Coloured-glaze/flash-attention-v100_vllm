
time=$(date '+%F_%H%M')
git_commit=$(git rev-parse --short HEAD)

echo "git commit: $(git rev-parse HEAD)" > profile_out_${time}_${git_commit}.txt

echo "start benchmark at ${time}" \
&& \
compute-sanitizer --print-limit 1 python test_vllm_flash_attn.py --flops --flops_num 5 \
&& \
python test_vllm_flash_attn.py --flops --profile --use_sdpa >> profile_out_${time}_${git_commit}.txt \
&& \
echo "benchmark done at ${time}"

echo "start profile analysis"

KERNEL_REGEX='.*fwd.*'

ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:"${KERNEL_REGEX}" \
    -o "profile_out" \
    python test_vllm_flash_attn.py --flops \
&& \
ncu --import profile_out.ncu-rep --csv | head -n 200 > profile_out_${time}.csv \
&& \
python compact_ncu.py profile_out_${time}.csv >> profile_out_${time}_${git_commit}.txt \
&& \
echo "profile analysis done, result saved to profile_out_${time}_${git_commit}.txt"


