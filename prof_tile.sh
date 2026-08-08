
cd ./tests/ncu_analyse/

time=$(date '+%y%m%d_%H%M')
git_commit=$(git rev-parse --short HEAD)
prefix_name="prof_tile"
output_file=${prefix_name}_${time}_${git_commit}.log

python test_vllm_flash_attn.py --flops --profile --use_tile --use_sdpa > ${output_file} 

# KERNEL_REGEX='.*fwd.*'
KERNEL_REGEX='main_kernel'

ncu -f --target-processes all --set full \
    --kernel-name-base demangled \
    --kernel-name ::regex:"${KERNEL_REGEX}" \
    -o "profile_out" \
    python test_vllm_flash_attn.py --flops --use_tile \
&& \
ncu --import profile_out.ncu-rep --csv | head -n 200 > ${prefix_name}_${time}.csv \
&& \
python compact_ncu.py ${prefix_name}_${time}.csv >> ${output_file}\
&& \
rm ${prefix_name}_${time}.csv \
&& \
echo "profile analysis done. "
echo "result saved to ${PWD}/${output_file}"
