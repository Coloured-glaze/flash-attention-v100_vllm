
try:
    from vllm_flash_attn import flash_attn_func
except ImportError as e:
    raise ImportError("VLLM Flash Attention not found") from e

try:
    from tile_flash_attn import flashattn, ref_program, do_bench
except ImportError as e:
    raise ImportError("Tile Flash Attention not found") from e

import torch, math, os
import torch.nn.functional as F
# from torch.nn.attention import SDPBackend, sdpa_kernel
import argparse


def ref_program_fa(query, key, value):
    return flash_attn_func(
        query.permute(0,2,1,3), key.permute(0,2,1,3), value.permute(0,2,1,3)).permute(0,2,1,3)

def manual_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    att = torch.nn.functional.softmax(att, dim=-1)
    y = torch.matmul(att, v)
    return y


# region torch profile test
def torch_profile(
    args,
    BATCH = 1, N_HEADS = 16, SEQ_LEN = 4096, HEAD_DIM = 128,
    block_M=128, block_N=64, num_stages=1, threads=256, is_causal=False, attn_mask=False):

    causal, dtype, device = False, torch.float16, "cuda"
    torch.manual_seed(42)
    print(f"torch: {torch.__version__}, cudnn: {torch.backends.cudnn.version()}, name: {torch.cuda.get_device_properties(0).name}")

    query = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    key = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    value = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, dtype=dtype, device=device)
    sm_scale = (1.0 / (HEAD_DIM ** 0.5))

    print("BATCH:", BATCH, "N_HEADS:", N_HEADS, "SEQ_LEN:", SEQ_LEN, "HEAD_DIM:", HEAD_DIM, "dtype:", dtype, "device:", device)

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
            line for line in prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=100).split('\n') if '---' not in line)
        print(filtered_output)
        print(result.shape, result[0][0][0][0:8], "\n")     
        return result

    # manual_result = profile_function("manual attention", manual_attn, query, key, value)
    
    vllm_flash_result = profile_function("vllm flash attention", ref_program_fa, query, key, value)

    tile_flash_result = None; sdp_result = None
    if args.use_tile:
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
    
    if args.use_sdpa:
        torch.backends.cuda.enable_math_sdp(False)  # use Torch
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # use xformers
        torch.backends.cuda.enable_flash_sdp(False)   # use flash attn backend
        # with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
        sdp_result = profile_function("sdpa", F.scaled_dot_product_attention, query, key, value, scale=sm_scale)

    if vllm_flash_result is not None and sdp_result is not None:
        diff = torch.abs(vllm_flash_result - sdp_result)
        print(f"VLLM Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n", flush=True)
        if not torch.allclose(vllm_flash_result, sdp_result, atol=5e-4, rtol=5e-4):
            print("VLLM Flash Attention and SDPA results do not match.")

    if tile_flash_result is not None and sdp_result is not None:
        diff = torch.abs(tile_flash_result - sdp_result)
        print(f"Tile Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n", flush=True)
        if not torch.allclose(tile_flash_result, sdp_result, atol=5e-4, rtol=5e-4):
            print("Tile Flash Attention and SDPA results do not match.")

# ==========================================================================================

# region flops test
def flops(
    args,
    is_causal: bool = False, attn_mask: bool = False, num: int = 100, 
    BATCH = 1, N_HEADS = 16, SEQ_LEN = 4096, HEAD_DIM = 128,
    block_M=128, block_N=64, num_stages=2, threads=128
):
    q_shape = (BATCH, N_HEADS, SEQ_LEN, HEAD_DIM) 
    kv_shape = (BATCH, N_HEADS, SEQ_LEN, HEAD_DIM) 

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
    
    print("BATCH:", q_shape[0], "N_HEADS:", q_shape[1], "SEQ_LEN:", q_shape[2], "HEAD_DIM:", q_shape[3], "dtype:", dtype, "device:", device)
    print("KV_BATCH:", kv_shape[0], "KV_N_HEADS:", kv_shape[1], "KV_SEQ_LEN:", kv_shape[2], "KV_HEAD_DIM:", kv_shape[3])

    fa_result = ref_program_fa(query, key, value)

    ref_result = None; tile_result = None
    if args.use_sdpa:
        ref_result = ref_program(query, key, value, mask, is_causal)
    
    if args.use_tile:
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
            with open(os.path.join(os.path.dirname(__file__), "tile_flash_attention.cu"), "w") as f:
                f.write(kernel.get_kernel_source())
    
    if ref_result is not None:
        diff = torch.abs(fa_result - ref_result)
        print(f"VLLM Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n")
        if not torch.allclose(fa_result, ref_result, atol=5e-4, rtol=5e-4):
            print("VLLM Flash Attention and Ref torch results do not match.")
    
    if tile_result is not None and ref_result is not None:
        diff = torch.abs(tile_result - ref_result)
        print(f"Tile Flash Attention diff max: {diff.max()}, mean: {diff.mean()}\n")
        if not torch.allclose(tile_result, ref_result, atol=5e-4, rtol=5e-4):
            print("Tile Flash Attention and Ref torch results do not match.")

    latency = do_bench(lambda: ref_program_fa(query, key, value), warmup=num)
    print("VLLM Flash Attention: {:.3f} ms".format(latency))
    print("VLLM Flash Attention: {:.3f} TFlops \n".format(total_flops / latency * 1e-9))

    if args.use_tile:
        if attn_mask:
            latency = do_bench(lambda: kernel(query, key, value, mask), warmup=num)
        else:   
            latency = do_bench(lambda: kernel(query, key, value), warmup=num)
        print("Tile Flash Attention: {:.3f} ms".format(latency))
        print("Tile Flash Attention: {:.3f} TFlops \n".format(total_flops / latency * 1e-9))
    
    if args.use_sdpa:
        latency = do_bench(lambda: ref_program(query, key, value, mask, is_causal), warmup=num)
        print("Ref SDPA: {:.3f} ms".format(latency))
        print("Ref SDPA: {:.3f} TFlops \n".format(total_flops / latency * 1e-9))
        torch.testing.assert_close(fa_result, ref_result, rtol=5e-4, atol=5e-4)
        print(f"fa_result checks pass.")
    
    if args.use_tile and ref_result is not None:
        torch.testing.assert_close(tile_result, ref_result, rtol=5e-4, atol=5e-4)
        print(f"tile_result checks pass.")


# region run test

if __name__ == "__main__":
    import os

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    
    parser = argparse.ArgumentParser(description="Test VLLM Flash Attention")
    parser.add_argument("--profile", action="store_true", help="Profile the kernel")
    parser.add_argument("--flops", action="store_true", help="Test the flops")
    parser.add_argument("--show_tile_source", action="store_true", help="Show the tile source")
    parser.add_argument("--use_tile", action="store_true", help="Use the tile flash attention")
    parser.add_argument("--use_sdpa", action="store_true", help="Use the SDPA results")
    parser.add_argument("--flops_num", type=int, default=200, help="Number of flops test")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--n_heads", type=int, default=40, help="Number of heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    
    args = parser.parse_args()
    if args.profile:
        torch_profile(
            args,
            BATCH = args.batch, N_HEADS = args.n_heads, SEQ_LEN = args.seq_len, HEAD_DIM = args.head_dim,
            block_M=128, block_N=64, num_stages=1, threads=256, is_causal=False, attn_mask=False
            )

    if args.flops:
        flops(
            args, num = args.flops_num,
            BATCH = args.batch, N_HEADS = args.n_heads, SEQ_LEN = args.seq_len, HEAD_DIM = args.head_dim,
            block_M=128, block_N=64, num_stages=1, threads=256, is_causal=False, attn_mask=False
        )
