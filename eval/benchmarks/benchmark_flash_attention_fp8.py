import argparse

import torch

from eval.common import (
    add_common_benchmark_args,
    error_metrics,
    print_backend_result,
    require_cuda,
    selected_dtypes,
    selected_shapes,
    time_cuda,
)
from kernels.flash_attention_fp8 import (
    flash_attention_fp8_cuda,
    flash_attention_fp8_pytorch,
    flash_attention_fp8_triton,
)


DEFAULT_SHAPES = [(1, 4, 128, 64), (1, 8, 256, 64), (1, 4, 256, 128)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark fake-FP8 causal FlashAttention backends.")
    add_common_benchmark_args(parser, dtypes=("float16",))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()
    if torch.cuda.get_device_capability(0)[0] < 9:
        print("note: this GPU uses the uint8 E4M3 educational path, not native FP8 Tensor Cores")

    for shape in selected_shapes(args.shape, DEFAULT_SHAPES):
        if len(shape) != 4:
            raise ValueError(f"FlashAttention shape must be BxHxSxD, got {shape}")
        for dtype in selected_dtypes(args.dtype, defaults=("float16",)):
            torch.manual_seed(8)
            q = torch.randn(shape, device="cuda", dtype=dtype)
            k = torch.randn(shape, device="cuda", dtype=dtype)
            v = torch.randn(shape, device="cuda", dtype=dtype)
            expected = flash_attention_fp8_pytorch(q, k, v)
            pytorch_ms = time_cuda(
                lambda: flash_attention_fp8_pytorch(q, k, v), warmup=args.warmup, repeat=args.repeat
            )

            print(f"\noperator=flash_attention_fp8 shape={shape} dtype={dtype}")
            print_backend_result("pytorch", pytorch_ms, reference_ms=pytorch_ms, metrics=error_metrics(expected, expected))
            for name, fn in (
                ("triton", flash_attention_fp8_triton),
                ("cuda", flash_attention_fp8_cuda),
            ):
                actual = fn(q, k, v)
                assert torch.allclose(actual, expected, rtol=3e-2, atol=8e-2)
                latency_ms = time_cuda(lambda fn=fn: fn(q, k, v), warmup=args.warmup, repeat=args.repeat)
                print_backend_result(name, latency_ms, reference_ms=pytorch_ms, metrics=error_metrics(actual, expected))


if __name__ == "__main__":
    main()

