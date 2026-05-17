import platform; 
system= platform.system().lower()

try:
    from flash_attn import flash_attn_func 
    # from vllm_flash_attn import flash_attn_func 
except ImportError as e:
    raise ImportError("VLLM Flash Attention not found") from e

if system == "linux":
    try:
        from tile_flash_attn import flashattn, do_bench 
    except ImportError as e:
        tilelang_source = """
        https://github.com/tile-ai/tilelang/
        https://github.com/tile-ai/tilelang/releases/download/v0.1.8/tilelang-0.1.8-cp38-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl
        """
        print(f"Tile Flash Attention not found, {e}")
        print(f"Tilelang source: \n{tilelang_source}")
        from triton.testing import do_bench
else :
    from triton.testing import do_bench

import torch, math, os
import torch.nn.functional as F
# from torch.nn.attention import SDPBackend, sdpa_kernel
import argparse
import matplotlib.pyplot as plt


def ref_program(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, is_causal: bool = False):
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, 
        attn_mask=mask, dropout_p=0.0, is_causal=is_causal)

def ref_program_fa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal=False):
    return flash_attn_func(
        query.permute(0,2,1,3), key.permute(0,2,1,3), value.permute(0,2,1,3),
        causal=causal
        ).permute(0,2,1,3)

def manual_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = torch.nn.functional.softmax(att, dim=-1)
    y = torch.matmul(att, v)
    return y

# def benchmark(f, *args, **kwargs):
#     return (do_bench(lambda: f(*args, **kwargs), return_mode="mean") * 1e3)  # return in us

# region torch profile test
def torch_profile(
    args, q_shape, kv_shape,
    block_M=128, block_N=64, num_stages=1, threads=256, is_causal=False, attn_mask=False):

    causal, dtype, device = False, torch.float16, "cuda"
    torch.manual_seed(42)
    print(f"torch: {torch.__version__}, cudnn: {torch.backends.cudnn.version()}, name: {torch.cuda.get_device_properties(0).name}")

    query = torch.randn(q_shape, dtype=dtype, device=device)
    key = torch.randn(kv_shape, dtype=dtype, device=device)
    value = torch.randn(kv_shape, dtype=dtype, device=device)
    sm_scale = (1.0 / (q_shape[3] ** 0.5))

    print("Q BATCH:", query.shape[0], "Q N_HEADS:", query.shape[1], "Q SEQ_LEN:", query.shape[2], "Q HEAD_DIM:", query.shape[3], "dtype:", dtype, "device:", device)
    print("kV BATCH:", key.shape[0], "kV N_HEADS:", key.shape[1], "kV SEQ_LEN:", key.shape[2], "kV HEAD_DIM:", key.shape[3])

    def profile_function(name, func, *args, **kwargs):
        for i in range(5): 
            _ = func(*args, **kwargs); torch.cuda.synchronize()
        print(f'=== profiling {name} ===')

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True, profile_memory=True, # use_cuda=True,
        ) as prof:
            for _ in range(5):
                func(*args, **kwargs); torch.cuda.synchronize()
            
            with torch.profiler.record_function(f"{name}_event"):
                result = func(*args, **kwargs); torch.cuda.synchronize()
        torch.cuda.empty_cache()
        filtered_output = '\n'.join(
            line for line in prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=200).split('\n') if '---' not in line)
        print(filtered_output)
        print(result.shape, result[0][0][0][0:8], "\n")     
        return result

    # manual_result = profile_function("manual attention", manual_attn, query, key, value)
    
    tile_flash_result = None; sdpa_result = None; vllm_flash_result = None
    if args.fa:
        vllm_flash_result = profile_function("vllm flash attention", ref_program_fa, query, key, value)
    if args.tile:
        kernel = flashattn(
        query.shape,
        key.shape,
        is_causal=is_causal,
        attn_mask=attn_mask,
        block_M=block_M,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads)
        tile_flash_result = profile_function("tile flash attention", kernel, query, key, value)
    
    if args.sdpa:
        torch.backends.cuda.enable_math_sdp(False)  # use Torch
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # use xformers
        torch.backends.cuda.enable_flash_sdp(False)   # use flash attn backend
        # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        sdpa_result = profile_function("sdpa", F.scaled_dot_product_attention, query, key, value, scale=sm_scale)

    if vllm_flash_result is not None and sdpa_result is not None:
        diff = torch.abs(vllm_flash_result - sdpa_result)
        print(f"VLLM Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n", flush=True)
        if not torch.allclose(vllm_flash_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("VLLM Flash Attention and SDPA results do not match.")

    if tile_flash_result is not None and sdpa_result is not None:
        diff = torch.abs(tile_flash_result - sdpa_result)
        print(f"Tile Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n", flush=True)
        if not torch.allclose(tile_flash_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("Tile Flash Attention and SDPA results do not match.")

# ==========================================================================================

# region flops test
plot_data = {
    'q_shapes': [],
    'kv_shapes': [],
    'fa_tflops': [],
    'fa_latency': [],
    'tile_tflops': [],
    'tile_latency': [],
    'sdpa_tflops': [],
    'sdpa_latency': [],
}

def flops(
    args, num, q_shape, kv_shape,
    block_M=128, block_N=64, num_stages=1, threads=256, 
    is_causal: bool = False, attn_mask: bool = False, 
):

    flops_per_matmul = 2.0 * q_shape[0] * q_shape[1] * q_shape[2] * kv_shape[2] * q_shape[3]
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    dtype, device = torch.float16, "cuda"
    torch.manual_seed(42)
    query = torch.randn(q_shape, dtype=dtype, device=device)
    key = torch.randn(kv_shape, dtype=dtype, device=device)
    value = torch.randn(kv_shape, dtype=dtype, device=device)
    
    if attn_mask:
        mask = torch.zeros(q_shape[0], q_shape[1], q_shape[2], kv_shape[2], dtype=dtype, device=device)
        mask[:, :, :, kv_shape[2]//2:] = -torch.inf # 屏蔽后半部分 keys
    else:
        mask = None
    
    print("Q BATCH:", q_shape[0], "Q N_HEADS:", q_shape[1], "Q SEQ_LEN:", q_shape[2], "Q HEAD_DIM:", q_shape[3], "dtype:", dtype, "device:", device)
    print("KV_BATCH:", kv_shape[0], "KV_N_HEADS:", kv_shape[1], "KV_SEQ_LEN:", kv_shape[2], "KV_HEAD_DIM:", kv_shape[3])

    sdpa_result = None; tile_result = None; fa_result = None
    if args.fa:
        fa_result = ref_program_fa(query, key, value)
    if args.sdpa:
        sdpa_result = ref_program(query, key, value, mask, is_causal)
    
    if args.tile:
        kernel = flashattn(
            q_shape,
            kv_shape,
            is_causal,
            attn_mask, 
            block_M=block_M,
            block_N=block_N,
            num_stages=num_stages,
            threads=threads)    
        
        if attn_mask:
            tile_result = kernel(query, key, value, mask)
        else:   
            tile_result = kernel(query, key, value)

        if args.show_tile_source:
            with open(os.path.join(os.path.dirname(__file__), f"tile_flash_attn_dim{q_shape[3]}.cu"), "w") as f:
                f.write(kernel.get_kernel_source())
                print(f"tile_flash_attn_dim{q_shape[3]}.cu saved.")
    
    if sdpa_result is not None and fa_result is not None:
        diff = torch.abs(fa_result - sdpa_result)
        print(f"VLLM Flash Attention Ref torch diff max: {diff.max()}, mean: {diff.mean()}\n")
        if not torch.allclose(fa_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("VLLM Flash Attention and Ref torch results do not match.")
    
    if tile_result is not None and sdpa_result is not None:
        diff = torch.abs(tile_result - sdpa_result)
        print(f"Tile Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n")
        if not torch.allclose(tile_result, sdpa_result, atol=5e-4, rtol=5e-4):
            print("Tile Flash Attention and Ref torch results do not match.")

    fa_latency = None; tile_latency = None; sdpa_latency = None
    fa_tflops = None; tile_tflops = None; sdpa_tflops = None
    
    if args.fa:
        fa_latency = do_bench(fn=lambda: ref_program_fa(query, key, value), rep=num, warmup=int(num/2))
        fa_tflops = total_flops / fa_latency * 1e-9
        print(f"VLLM Flash Attention: {fa_latency:.3f} ms")
        print(f"VLLM Flash Attention: {fa_tflops:.3f} TFlops")

    if args.tile:
        if attn_mask:
            tile_latency = do_bench(fn=lambda: kernel(query, key, value, mask), rep=num, warmup=int(num/2))
        else:   
            tile_latency = do_bench(fn=lambda: kernel(query, key, value), rep=num, warmup=int(num/2))
        tile_tflops = total_flops / tile_latency * 1e-9
        print(f"Tile Flash Attention: {tile_latency:.3f} ms")
        print(f"Tile Flash Attention: {tile_tflops:.3f} TFlops")
    
    if args.sdpa :
        sdpa_latency = do_bench(fn=lambda: ref_program(query, key, value, mask, is_causal), rep=num, warmup=int(num/2))
        sdpa_tflops = total_flops / sdpa_latency * 1e-9
        print(f"Ref SDPA: {sdpa_latency:.3f} ms")
        print(f"Ref SDPA: {sdpa_tflops:.3f} TFlops")

    if (fa_result is not None) and (sdpa_result is not None) :
        vllm_fa_speed_up = (1 - (float(fa_latency) / float(sdpa_latency)) ) * 100
        torch.testing.assert_close(fa_result, sdpa_result, rtol=5e-4, atol=5e-4)
        print(f"fa result checks pass. fa speed up: {vllm_fa_speed_up:.2f}%\n")
    
    if (tile_result is not None) and (sdpa_result is not None) :
        tile_speed_up = (1 - (float(tile_latency) / float(sdpa_latency)) ) * 100
        torch.testing.assert_close(tile_result, sdpa_result, rtol=5e-4, atol=5e-4)
        print(f"tile result checks pass. tile speed up: {tile_speed_up:.2f}%\n")
    
    if args.plot:
        plot_data['q_shapes'].append(q_shape)
        plot_data['kv_shapes'].append(kv_shape)
        if fa_tflops is not None:
            plot_data['fa_tflops'].append(fa_tflops)
            plot_data['fa_latency'].append(fa_latency)
        if tile_tflops is not None:
            plot_data['tile_tflops'].append(tile_tflops)
            plot_data['tile_latency'].append(tile_latency)
        if sdpa_tflops is not None:
            plot_data['sdpa_tflops'].append(sdpa_tflops)
            plot_data['sdpa_latency'].append(sdpa_latency)


# region run test
if __name__ == "__main__":
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    parser = argparse.ArgumentParser(description="Test VLLM Flash Attention")
    parser.add_argument("--profile", action="store_true", help="Profile the kernel")
    parser.add_argument("--flops", action="store_true", help="Test the flops")
    parser.add_argument("--flops_num", type=int, default=500, help="Number of flops test")
    parser.add_argument("--show_tile_source", action="store_true", help="Show the tilelang cuda source code")

    parser.add_argument("--fa", action="store_true", help="Use the flash attention")
    parser.add_argument("--sdpa", action="store_true", help="Use the SDPA results")
    parser.add_argument("--tile", action="store_true", help="Use the tile flash attention")

    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--n_heads", type=int, default=40, help="Number of heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    
    parser.add_argument("--plot", action="store_true", help="Plot q_shape, kv_shape, tflops and time charts when running flops test")
    
    args = parser.parse_args()

    # q_shape = (args.batch, args.n_heads, args.seq_len, args.head_dim) 
    # kv_shape = (args.batch, args.n_heads, args.seq_len, args.head_dim) 

    q_shape_list = []
    kv_shape_list = []

    # q_shape_list.append((1, 40, 1, args.head_dim) )
    # kv_shape_list.append((1, 40, 512, args.head_dim) )

    q_shape_list.append((1, 40, 1, args.head_dim))
    kv_shape_list.append((1, 40, 65536, args.head_dim))

    q_shape_list.append((1, 40, 1, args.head_dim))
    kv_shape_list.append((1, 40, 131072, args.head_dim))

    # q_shape_list.append((2, 20, 1024, args.head_dim))
    # kv_shape_list.append((2, 20, 77, args.head_dim))

    q_shape_list.append((2, 10, 4096, args.head_dim))
    kv_shape_list.append((2, 10, 77, args.head_dim))

    # q_shape_list.append((2, 10, 6144, args.head_dim))
    # kv_shape_list.append((2, 10, 77, args.head_dim))

    q_shape_list.append((2, 16, 3952,  args.head_dim))
    kv_shape_list.append((2, 16, 512,  args.head_dim))

    q_shape_list.append((2, 16, 3952,  args.head_dim))
    kv_shape_list.append((2, 16, 3952,  args.head_dim))

    # q_shape_list.append((8, 10, 512, args.head_dim))
    # kv_shape_list.append((8, 10, 512, args.head_dim))

    # q_shape_list.append((8, 10, 1024, args.head_dim))
    # kv_shape_list.append((8, 10, 1024, args.head_dim))

    # q_shape_list.append((8, 10, 2048, args.head_dim))
    # kv_shape_list.append((8, 10, 2048, args.head_dim))

    # q_shape_list.append((2, 40, 4096, args.head_dim))
    # kv_shape_list.append((2, 40, 4096, args.head_dim))

    q_shape_list.append((2, 40, 8192, args.head_dim))
    kv_shape_list.append((2, 40, 8192, args.head_dim))

    # q_shape_list.append((2, 40, 16384, args.head_dim))
    # kv_shape_list.append((2, 40, 16384, args.head_dim))

    for q_shape, kv_shape in zip(q_shape_list, kv_shape_list):
        if args.profile:
            torch_profile(
                args,
                q_shape, kv_shape,
                block_M=128, block_N=64, num_stages=1, threads=256, is_causal=False, attn_mask=False
                )

        if args.flops:
            flops(
                args, is_causal=False, attn_mask=False, num = args.flops_num,
                q_shape=q_shape, kv_shape=kv_shape,
                block_M=128, block_N=64, num_stages=1, threads=256, 
            )
    
    if args.flops and args.plot and plot_data['q_shapes']:
        print("\n=== Generating plots ===")
        x_labels_q = [str(tuple(q)) for q in plot_data['q_shapes']]
        x_labels_kv = [str(tuple(kv)) for kv in plot_data['kv_shapes']]
        x_labels = [f"{q}\n{kv}" for q, kv in zip(x_labels_q, x_labels_kv)]
        x = range(len(x_labels))
        
        # 根据数据点数量动态调整柱宽和图表大小
        num_points = len(x_labels)
        width = 0.8 / (len([k for k in ['fa_tflops', 'tile_tflops', 'sdpa_tflops'] if plot_data[k]]) + 1)
        figsize_width = min(18, max(12, num_points * 2.5))
        
        colors = {
            'fa': '#1f77b4',      # blue
            'tile': '#ff7f0e',    # orange
            'sdpa': '#2ca02c',    # green
        }
        
        # 动态调整字体大小
        label_fontsize = max(6, min(9, 12 - num_points))
        value_fontsize = max(5, min(8, 10 - num_points))
        
        fig, axes = plt.subplots(1, 2, figsize=(figsize_width, 6))
        
        # TFLOPs chart
        ax1 = axes[0]
        offset1 = 0
        if plot_data['fa_tflops']:
            rects = ax1.bar([i + offset1 for i in x], plot_data['fa_tflops'], width, 
                           label='FlashAttention', color=colors['fa'], edgecolor='black', linewidth=0.5)
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
            offset1 += width
        if plot_data['tile_tflops']:
            rects = ax1.bar([i + offset1 for i in x], plot_data['tile_tflops'], width,
                           label='TileLang', color=colors['tile'], edgecolor='black', linewidth=0.5)
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
            offset1 += width
        if plot_data['sdpa_tflops']:
            rects = ax1.bar([i + offset1 for i in x], plot_data['sdpa_tflops'], width,
                           label='SDPA', color=colors['sdpa'], edgecolor='black', linewidth=0.5)
            for rect in rects:
                height = rect.get_height()
                ax1.annotate(f'{height:.1f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
        
        ax1.set_ylabel('Speed (TFLOP/s)', fontsize=12)
        ax1.set_xlabel('Sequence length', fontsize=12)
        ax1.set_title(f'v100 Attention forward TFLOPs hdim {args.head_dim}', fontsize=13)
        ax1.set_xticks([i + width/2 for i in x])
        ax1.set_xticklabels(x_labels, fontsize=label_fontsize, rotation=0)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_axisbelow(True)
        ax1.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Latency chart
        ax2 = axes[1]
        offset2 = 0
        if plot_data['fa_latency']:
            rects = ax2.bar([i + offset2 for i in x], plot_data['fa_latency'], width,
                           label='FlashAttention', color=colors['fa'], edgecolor='black', linewidth=0.5)
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
            offset2 += width
        if plot_data['tile_latency']:
            rects = ax2.bar([i + offset2 for i in x], plot_data['tile_latency'], width,
                           label='TileLang', color=colors['tile'], edgecolor='black', linewidth=0.5)
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
            offset2 += width
        if plot_data['sdpa_latency']:
            rects = ax2.bar([i + offset2 for i in x], plot_data['sdpa_latency'], width,
                           label='SDPA', color=colors['sdpa'], edgecolor='black', linewidth=0.5)
            for rect in rects:
                height = rect.get_height()
                ax2.annotate(f'{height:.3f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                           ha='center', va='bottom', fontsize=value_fontsize, xytext=(0, 2), textcoords='offset points')
        
        ax2.set_ylabel('Latency (ms)', fontsize=12)
        ax2.set_xlabel('Sequence length', fontsize=12)
        ax2.set_title(f'v100 Attention latency hdim {args.head_dim}', fontsize=13)
        ax2.set_xticks([i + width/2 for i in x])
        ax2.set_xticklabels(x_labels, fontsize=label_fontsize, rotation=0)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.set_axisbelow(True)
        ax2.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        save_plot = f'flops_{str(q_shape_list[-1]).replace(" ", "")}_{str(kv_shape_list[-1]).replace(" ", "")}.png'
        plt.tight_layout(); 
        plt.savefig(save_plot, dpi=150, bbox_inches='tight')
        print(f"Plot saved as {save_plot}")
        plt.show()


Pytorch_AttentionKernel_Generator = """
    def cpp_class(self) -> str:
        template_args = ", ".join(
            [
                DTYPES[self.dtype],
                f"cutlass::arch::Sm{self.sm_range[0]}",
                "true" if self.aligned else "false",
                str(self.q),
                str(self.k),
                str(self.max_k),
                "true" if self.supports_dropout else "false",
                "true" if self.supports_bias else "false",
            ]
        )
        return f"AttentionKernel<{template_args}>"
"""

# python tests/ncu_analyse/test_vllm_flash_attn.py --flops --fa --sdpa --head_dim 128 
# python tests/ncu_analyse/test_vllm_flash_attn.py --flops --profile --fa --sdpa --head_dim 64
