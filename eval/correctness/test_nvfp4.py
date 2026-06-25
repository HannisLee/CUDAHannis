import pytest
import torch

from kernels.nvfp4 import (
    nvfp4_dequantize_pytorch,
    nvfp4_quantize_cuda,
    nvfp4_quantize_dequantize_cuda,
    nvfp4_quantize_dequantize_pytorch,
    nvfp4_quantize_dequantize_triton,
    nvfp4_quantize_pytorch,
    nvfp4_quantize_triton,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for custom kernels.")


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 16), (3, 17), (2, 5, 33)])
def test_nvfp4_float32_and_float16(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    torch.manual_seed(4)
    x = torch.randn(*shape, device="cuda", dtype=dtype) * 3.0
    _assert_matches_reference(x)


def test_nvfp4_known_codes_and_padding() -> None:
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
