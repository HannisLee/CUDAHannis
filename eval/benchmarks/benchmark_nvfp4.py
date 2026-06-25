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
from kernels.nvfp4 import (
    nvfp4_dequantize_pytorch,
    nvfp4_quantize_cuda,
    nvfp4_quantize_pytorch,
    nvfp4_quantize_triton,
)


DEFAULT_SHAPES = [(1, 128, 4096), (1, 128, 12288)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark NVFP4-style quantization backends.")
    add_common_benchmark_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_cuda()
    if torch.cuda.get_device_capability(0)[0] < 10:
        print("note: this is the educational NVFP4-style path, not native Blackwell NVFP4 Tensor Core execution")

    for shape in selected_shapes(args.shape, DEFAULT_SHAPES):
        for dtype in selected_dtypes(args.dtype):
            torch.manual_seed(4)
            x = torch.randn(shape, device="cuda", dtype=dtype) * 3.0
            packed_ref, scales_ref = nvfp4_quantize_pytorch(x)
            reference = nvfp4_dequantize_pytorch(packed_ref, scales_ref, x.shape[-1], out_dtype=x.dtype)
            pytorch_ms = time_cuda(lambda: nvfp4_quantize_pytorch(x), warmup=args.warmup, repeat=args.repeat)

            print(f"\noperator=nvfp4 shape={shape} dtype={dtype}")
            reference_metrics = error_metrics(reference, x)
            reference_metrics["reconstruction_mse"] = float(torch.mean((reference.float() - x.float()) ** 2).item())
            print_backend_result("pytorch", pytorch_ms, reference_ms=pytorch_ms, metrics=reference_metrics)

            for name, fn in (("triton", nvfp4_quantize_triton), ("cuda", nvfp4_quantize_cuda)):
                packed, scales = fn(x)
                actual = nvfp4_dequantize_pytorch(packed, scales, x.shape[-1], out_dtype=x.dtype)
                packed_mismatches = int((packed != packed_ref).sum().item())
                assert packed_mismatches == 0
                assert torch.allclose(scales, scales_ref, rtol=1e-6, atol=1e-6)
                assert torch.allclose(actual, reference, rtol=1e-5, atol=1e-5)

                latency_ms = time_cuda(lambda fn=fn: fn(x), warmup=args.warmup, repeat=args.repeat)
                metrics = error_metrics(actual, reference)
                metrics.update(
                    {
                        "packed_mismatch_ratio": packed_mismatches / packed_ref.numel(),
                        "scale_max_abs_error": float((scales - scales_ref).abs().max().item()),
                        "reconstruction_mse": float(torch.mean((actual.float() - x.float()) ** 2).item()),
                    }
                )
                print_backend_result(name, latency_ms, reference_ms=pytorch_ms, metrics=metrics)


if __name__ == "__main__":
    main()

