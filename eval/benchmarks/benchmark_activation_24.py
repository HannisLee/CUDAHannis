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
from kernels.activation_24 import (
    activation_24_sparsity_cuda,
    activation_24_sparsity_pytorch,
    activation_24_sparsity_triton,
)


DEFAULT_SHAPES = [(1, 128, 4096), (1, 128, 12288)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark activation 2:4 sparsity backends.")
    add_common_benchmark_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()

    for shape in selected_shapes(args.shape, DEFAULT_SHAPES):
        for dtype in selected_dtypes(args.dtype):
            torch.manual_seed(24)
            x = torch.randn(shape, device="cuda", dtype=dtype)
            expected = activation_24_sparsity_pytorch(x)
            pytorch_ms = time_cuda(
                lambda: activation_24_sparsity_pytorch(x), warmup=args.warmup, repeat=args.repeat
            )

            print(f"\noperator=activation_24 shape={shape} dtype={dtype}")
            print_backend_result("pytorch", pytorch_ms, reference_ms=pytorch_ms, metrics=error_metrics(expected, expected))
            for name, fn in (
                ("triton", activation_24_sparsity_triton),
                ("cuda", activation_24_sparsity_cuda),
            ):
                actual = fn(x)
                assert torch.equal(actual, expected)
                latency_ms = time_cuda(lambda fn=fn: fn(x), warmup=args.warmup, repeat=args.repeat)
                print_backend_result(name, latency_ms, reference_ms=pytorch_ms, metrics=error_metrics(actual, expected))


if __name__ == "__main__":
    main()

