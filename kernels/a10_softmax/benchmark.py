import sys
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from softmax_pytorch import softmax_pytorch
from softmax_triton import softmax_triton

lib = load(
    name="softmax_lib",
    sources=[str(CURRENT_DIR / "softmax.cu")],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)


def run_benchmark(
    perf_func: Callable,
    x: torch.Tensor,
    tag: str,
    ref: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 100,
    show_all: bool = False,
):
    has_out = out is not None

    if has_out:
        for _ in range(warmup):
            perf_func(x, out)
    else:
        for _ in range(warmup):
            out = perf_func(x)

    torch.cuda.synchronize()
    start = time.perf_counter()

    if has_out:
        for _ in range(iters):
            perf_func(x, out)
    else:
        for _ in range(iters):
            out = perf_func(x)

    torch.cuda.synchronize()
    end = time.perf_counter()

    assert out is not None
    mean_time = (end - start) * 1000 / iters
    out_val = out.flatten().detach().cpu().tolist()[:3]
    out_val = [round(float(v), 8) for v in out_val]
    out_val = [f"{v:<12}" for v in out_val]

    if ref is None:
        max_abs = 0.0
    else:
        max_abs = (out - ref).abs().max().item()

    print(f"{('out_' + tag):>22}: {out_val}, time:{mean_time:.8f}ms, max_abs:{max_abs:.3e}")

    if show_all:
        print(out)

    return out, mean_time


# (kernel, tag, 支持的 H 列表) —— H 列表与 softmax.cu 中各 dispatch 宏的 switch case 对应
FP32_KERNELS: list[Tuple[Callable, str, list]] = [
    (lib.softmax_f32_per_token, "f32", [32, 64, 128, 256, 512, 1024]),
    (lib.softmax_f32x4_per_token, "f32x4", [32, 64, 128, 256, 512, 1024, 2048, 4096]),
    (lib.safe_softmax_f32_per_token, "safe_f32", [32, 64, 128, 256, 512, 1024]),
    (lib.safe_softmax_f32x4_per_token, "safe_f32x4", [32, 64, 128, 256, 512, 1024, 2048, 4096]),
    (lib.online_safe_softmax_f32_per_token, "online_f32", [32, 64, 128, 256, 512, 1024]),
    (
        lib.online_safe_softmax_f32x4_pack_per_token,
        "online_f32x4",
        [128, 256, 512, 1024, 2048, 4096],
    ),
]

FP16_KERNELS: list[Tuple[Callable, str, list]] = [
    (lib.safe_softmax_f16_f32_per_token, "f16_f32", [32, 64, 128, 256, 512, 1024]),
    (
        lib.safe_softmax_f16x2_f32_per_token,
        "f16x2_f32",
        [32, 64, 128, 256, 512, 1024, 2048],
    ),
    (
        lib.safe_softmax_f16x8_pack_f32_per_token,
        "f16x8_pack",
        [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    ),
]


def bench_shape(S: int, H: int, dtype: torch.dtype):
    dtype_str = "fp32" if dtype == torch.float32 else "fp16"
    print("-" * 105)
    print(" " * 32 + f"S={S}, H={H}, dtype={dtype_str}")
    print("-" * 105)

    x = torch.randn((S, H), device="cuda", dtype=dtype).contiguous()
    out = torch.empty((S, H), device="cuda", dtype=dtype)

    ref, _ = run_benchmark(softmax_pytorch, x, "torch")

    kernels = FP32_KERNELS if dtype == torch.float32 else FP16_KERNELS
    for func, tag, hs in kernels:
        if H in hs:
            run_benchmark(func, x, tag, ref, out)
        else:
            print(f"{('out_' + tag):>22}: [skip: H={H} not supported by this kernel]")

    run_benchmark(softmax_triton, x, "triton", ref)

    print("-" * 105)


if __name__ == "__main__":
    # fp32: 6 个 kernel 全部支持 H ∈ {128, 256, 512, 1024}
    for H in [128, 256, 512, 1024]:
        bench_shape(4096, H, torch.float32)

    # fp16: 3 个 kernel 全部支持 H ∈ {128, 256, 512, 1024}
    for H in [128, 256, 512, 1024]:
        bench_shape(4096, H, torch.float16)

    # fp16 大 H: 仅 f16x2 / f16x8_pack 支持，体现向量化在大 H 下的扩展性
    for H in [2048, 4096, 8192]:
        bench_shape(4096, H, torch.float16)
