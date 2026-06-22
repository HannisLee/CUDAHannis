import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.nvfp4 import (
    nvfp4_dequantize_pytorch,
    nvfp4_quantize_cuda,
    nvfp4_quantize_dequantize_cuda,
    nvfp4_quantize_dequantize_pytorch,
    nvfp4_quantize_dequantize_triton,
    nvfp4_quantize_pytorch,
    nvfp4_quantize_triton,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请检查 nvidia-smi、driver、PyTorch CUDA wheel 和环境激活状态。")


def _assert_matches_reference(x: torch.Tensor) -> None:
    packed_ref, scales_ref = nvfp4_quantize_pytorch(x)
    packed_triton, scales_triton = nvfp4_quantize_triton(x)
    packed_cuda, scales_cuda = nvfp4_quantize_cuda(x)

    assert torch.equal(packed_triton, packed_ref)
    assert torch.equal(packed_cuda, packed_ref)
    assert torch.allclose(scales_triton, scales_ref, rtol=1e-6, atol=1e-6)
    assert torch.allclose(scales_cuda, scales_ref, rtol=1e-6, atol=1e-6)

    ref = nvfp4_dequantize_pytorch(packed_ref, scales_ref, x.shape[-1], out_dtype=x.dtype)
    assert torch.allclose(nvfp4_quantize_dequantize_pytorch(x), ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(nvfp4_quantize_dequantize_triton(x), ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(nvfp4_quantize_dequantize_cuda(x), ref, rtol=1e-5, atol=1e-5)


def test_nvfp4_float32_and_float16() -> None:
    _require_cuda()
    torch.manual_seed(4)
    for dtype in (torch.float32, torch.float16):
        for shape in ((2, 16), (3, 17), (2, 5, 33)):
            x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0
            _assert_matches_reference(x)


def test_nvfp4_known_codes_and_padding() -> None:
    _require_cuda()
    x = torch.tensor(
        [[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, 0.0, 1.0]],
        device="cuda",
        dtype=torch.float32,
    )
    packed, scales = nvfp4_quantize_pytorch(x)
    assert packed.shape == (1, 16)
    assert scales.shape == (1, 2)
    assert torch.equal(nvfp4_quantize_triton(x)[0], packed)
    assert torch.equal(nvfp4_quantize_cuda(x)[0], packed)


def main() -> None:
    _require_cuda()
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(4, 16, 4099, device="cuda", dtype=dtype) * 3.0
        packed_ref, scales_ref = nvfp4_quantize_pytorch(x)
        packed_triton, scales_triton = nvfp4_quantize_triton(x)
        packed_cuda, scales_cuda = nvfp4_quantize_cuda(x)
        ref = nvfp4_dequantize_pytorch(packed_ref, scales_ref, x.shape[-1], out_dtype=x.dtype)
        triton_out = nvfp4_dequantize_pytorch(packed_triton, scales_triton, x.shape[-1], out_dtype=x.dtype)
        cuda_out = nvfp4_dequantize_pytorch(packed_cuda, scales_cuda, x.shape[-1], out_dtype=x.dtype)
        print(f"dtype: {dtype}")
        print(f"packed shape: {tuple(packed_ref.shape)}")
        print(f"scales shape: {tuple(scales_ref.shape)}")
        print(f"triton max error: {(triton_out - ref).abs().max().item():.8e}")
        print(f"cuda max error: {(cuda_out - ref).abs().max().item():.8e}")
        assert torch.equal(packed_triton, packed_ref)
        assert torch.equal(packed_cuda, packed_ref)
        assert torch.allclose(triton_out, ref, rtol=1e-5, atol=1e-5)
        assert torch.allclose(cuda_out, ref, rtol=1e-5, atol=1e-5)
    print("NVFP4-style quantization correctness: PASS")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "CUDA_HOME" in str(exc) or "nvcc" in str(exc):
            pytest.skip(str(exc))
        raise
