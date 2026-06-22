import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.nvfp4 import (
    nvfp4_dequantize_pytorch,
    nvfp4_quantize_cuda,
    nvfp4_quantize_pytorch,
    nvfp4_quantize_triton,
)


def time_cuda(fn, warmup: int = 20, repeat: int = 100) -> float:
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
        raise RuntimeError("CUDA 不可用：请先运行 source scripts/activate_env.sh 并检查 GPU/driver/PyTorch CUDA。")

    capability = torch.cuda.get_device_capability(0)
    if capability[0] < 10:
        print(
            "note: 当前 GPU 不是 Blackwell/SM100+，本脚本运行的是 NVFP4-style pack/dequant 教学实现，"
            "不是原生 NVFP4 Tensor Core 路径。"
        )

    shape = (4, 16, 4099)
    print(f"input shape: {shape}")
    for dtype in (torch.float32, torch.float16):
        torch.manual_seed(4)
        x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0

        packed_ref, scales_ref = nvfp4_quantize_pytorch(x)
        packed_triton, scales_triton = nvfp4_quantize_triton(x)
        packed_cuda, scales_cuda = nvfp4_quantize_cuda(x)
        ref = nvfp4_dequantize_pytorch(packed_ref, scales_ref, x.shape[-1], out_dtype=x.dtype)
        triton_out = nvfp4_dequantize_pytorch(packed_triton, scales_triton, x.shape[-1], out_dtype=x.dtype)
        cuda_out = nvfp4_dequantize_pytorch(packed_cuda, scales_cuda, x.shape[-1], out_dtype=x.dtype)

        assert torch.equal(packed_triton, packed_ref)
        assert torch.equal(packed_cuda, packed_ref)
        assert torch.allclose(triton_out, ref, rtol=1e-5, atol=1e-5)
        assert torch.allclose(cuda_out, ref, rtol=1e-5, atol=1e-5)

        pytorch_ms = time_cuda(lambda: nvfp4_quantize_pytorch(x), warmup=10, repeat=30)
        triton_ms = time_cuda(lambda: nvfp4_quantize_triton(x))
        cuda_ms = time_cuda(lambda: nvfp4_quantize_cuda(x))
        mse = torch.mean((ref.float() - x.float()) ** 2).item()

        print(f"\ndtype: {dtype}")
        print(f"packed shape: {tuple(packed_ref.shape)}")
        print(f"scales shape: {tuple(scales_ref.shape)}")
        print(f"reconstruction mse: {mse:.8e}")
        print(f"Triton max error vs reference: {(triton_out - ref).abs().max().item():.8e}")
        print(f"CUDA max error vs reference: {(cuda_out - ref).abs().max().item():.8e}")
        print(f"PyTorch reference quantize: {pytorch_ms:.4f} ms")
        print(f"Triton quantize:           {triton_ms:.4f} ms")
        print(f"CUDA extension quantize:   {cuda_ms:.4f} ms")

    print("\nNVFP4-style single script: PASS")


if __name__ == "__main__":
    main()
