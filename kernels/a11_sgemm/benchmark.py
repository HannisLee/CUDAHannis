import time
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

lib = load(
    name="sgemm_lib",
    sources=["sgemm.cu"],
    extra_cuda_cflags=[
        "-O3",
        # parvati: nvcc 13.2 与 cuda-runtime headers 13.0 小版本不一致,
        # 禁用 CCCL 的 toolkit 兼容性强校验
        "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    extra_ldflags=["-lcublas", "-lcublasLt"],
)


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    tag: str,
    warmup: int = 5,
    iters: int = 10,
):
    c.fill_(0)  # 每次评测前清零,避免上一次结果残留

    # warmup
    for _ in range(warmup):
        perf_func(a, b, c, M, N, K)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        perf_func(a, b, c, M, N, K)
    torch.cuda.synchronize()
    end = time.time()

    mean_time = (end - start) * 1000 / iters

    # correctness vs torch reference(同一组 a/b)
    ref = a @ b
    max_diff = (c - ref).abs().max().item()
    out_val = c.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(float(v), 6) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]

    print(f"{tag:>12}: {out_val}, max_diff:{max_diff:.4e}, time:{mean_time:.6f}ms")
    return mean_time


def torch_sgemm(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    c.copy_(a @ b)


def bench_shape(M: int, N: int, K: int):
    print("-" * 85)
    print(" " * 40 + f"M={M}, N={N}, K={K}, dtype=fp32")
    print("-" * 85)

    # 同一组输入,三个实现共享 -> 输出值可直接对比
    a = torch.randn((M, K), device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn((K, N), device="cuda", dtype=torch.float32).contiguous()
    c = torch.zeros((M, N), device="cuda", dtype=torch.float32).contiguous()

    run_benchmark(lib.sgemm_v1, a, b, c, M, N, K, "sgemm_v1")
    run_benchmark(lib.sgemm_v2, a, b, c, M, N, K, "sgemm_v2")
    run_benchmark(lib.sgemm_v5, a, b, c, M, N, K, "sgemm_v5")
    run_benchmark(torch_sgemm, a, b, c, M, N, K, "torch")

    print("-" * 85)


if __name__ == "__main__":
    shapes = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
    for M, N, K in shapes:
        bench_shape(M, N, K)
