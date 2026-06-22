import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

QWEN35_9B_BATCH = 1
QWEN35_9B_SEQUENCE_LENGTH = 1024
QWEN35_9B_INTERMEDIATE_SIZE = 12288
CUDA_PACKED_MISMATCH_RATIO_LIMIT = 1e-5
CUDA_DEQUANT_MSE_LIMIT = 1e-6

from kernels.nvfp4 import (
    nvfp4_dequantize_pytorch,
    nvfp4_quantize_cuda,
    nvfp4_quantize_pytorch,
    nvfp4_quantize_triton,
)


def time_cuda(fn, warmup: int = 3, repeat: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeat


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    shape = (QWEN35_9B_BATCH, QWEN35_9B_SEQUENCE_LENGTH, QWEN35_9B_INTERMEDIATE_SIZE)
    print("operator: NVFP4-style quantize")
    print("model shape: Qwen3.5-9B MLP intermediate activation")
    print(f"shape: {shape}")

    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(4)
        x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0

        packed_ref, scales_ref = nvfp4_quantize_pytorch(x)
        packed_triton, scales_triton = nvfp4_quantize_triton(x)
        packed_cuda, scales_cuda = nvfp4_quantize_cuda(x)

        ref = nvfp4_dequantize_pytorch(packed_ref, scales_ref, x.shape[-1], out_dtype=x.dtype)
        triton_out = nvfp4_dequantize_pytorch(packed_triton, scales_triton, x.shape[-1], out_dtype=x.dtype)
        cuda_out = nvfp4_dequantize_pytorch(packed_cuda, scales_cuda, x.shape[-1], out_dtype=x.dtype)
        cuda_packed_mismatches = int((packed_cuda != packed_ref).sum().item())
        cuda_packed_mismatch_ratio = cuda_packed_mismatches / packed_ref.numel()
        cuda_dequant_mse = torch.mean((cuda_out.float() - ref.float()) ** 2).item()

        assert torch.equal(packed_triton, packed_ref)
        assert torch.allclose(scales_triton, scales_ref, rtol=1e-6, atol=1e-6)
        assert torch.allclose(scales_cuda, scales_ref, rtol=1e-6, atol=1e-6)
        assert torch.allclose(triton_out, ref, rtol=1e-5, atol=1e-5)
        assert cuda_packed_mismatch_ratio <= CUDA_PACKED_MISMATCH_RATIO_LIMIT
        assert cuda_dequant_mse <= CUDA_DEQUANT_MSE_LIMIT

        pytorch_ms = time_cuda(lambda: nvfp4_quantize_pytorch(x))
        triton_ms = time_cuda(lambda: nvfp4_quantize_triton(x))
        cuda_ms = time_cuda(lambda: nvfp4_quantize_cuda(x))

        print(f"\ndtype: {dtype}")
        print(f"packed shape:     {tuple(packed_ref.shape)}")
        print(f"scales shape:     {tuple(scales_ref.shape)}")
        print(f"triton max error: {(triton_out - ref).abs().max().item():.8e}")
        print(f"cuda max error:   {(cuda_out - ref).abs().max().item():.8e}")
        print(f"cuda packed diff: {cuda_packed_mismatches} / {packed_ref.numel()} ({cuda_packed_mismatch_ratio:.3e})")
        print(f"cuda dequant mse: {cuda_dequant_mse:.8e}")
        print(f"PyTorch:          {pytorch_ms:.4f} ms")
        print(f"Triton:           {triton_ms:.4f} ms")
        print(f"CUDA extension:   {cuda_ms:.4f} ms")

    print("\nPASS")


if __name__ == "__main__":
    main()
