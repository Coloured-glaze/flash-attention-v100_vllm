
echo "start benchmark"

python test_vllm_flash_attn.py --flops --profile --use_sdpa

echo "benchmark done"



echo "start profile analysis"

KERNEL_REGEX='.*fwd.*'

ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:"${KERNEL_REGEX}" \
    -o "profile_out" \
    python test_vllm_flash_attn.py --flops \
&& \
ncu --import profile_out.ncu-rep --csv | head -n 100 > profile_out.csv \
&& \
python compact_ncu.py profile_out.csv > profile_out.txt

echo "profile analysis done"


