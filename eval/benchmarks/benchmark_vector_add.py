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
from kernels.vector_add import cuda_add, triton_add


DEFAULT_SHAPES = [(1_000_003,), (16 * 1024 * 1024,)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark vector add backends.")
    add_common_benchmark_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()

    for shape in selected_shapes(args.shape, DEFAULT_SHAPES):
        for dtype in selected_dtypes(args.dtype):
            torch.manual_seed(1)
            x = torch.randn(shape, device="cuda", dtype=dtype)
            y = torch.randn(shape, device="cuda", dtype=dtype)
            expected = x + y

            pytorch_ms = time_cuda(lambda: x + y, warmup=args.warmup, repeat=args.repeat)
            print(f"\noperator=vector_add shape={shape} dtype={dtype}")
            print_backend_result("pytorch", pytorch_ms, reference_ms=pytorch_ms, metrics=error_metrics(expected, expected))

            for name, fn in (("triton", triton_add), ("cuda", cuda_add)):
                actual = fn(x, y)
                tolerance = 1e-3 if dtype == torch.float16 else 1e-6
                assert torch.allclose(actual, expected, rtol=tolerance, atol=tolerance)
                latency_ms = time_cuda(lambda fn=fn: fn(x, y), warmup=args.warmup, repeat=args.repeat)
                metrics = error_metrics(actual, expected)
                bytes_moved = 3 * x.numel() * x.element_size()
                metrics["bandwidth_gbps"] = bytes_moved / (latency_ms / 1_000) / 1e9
                print_backend_result(name, latency_ms, reference_ms=pytorch_ms, metrics=metrics)


if __name__ == "__main__":
    main()

