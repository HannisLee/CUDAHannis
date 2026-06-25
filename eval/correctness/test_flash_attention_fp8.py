import pytest
import torch

from kernels.flash_attention_fp8 import (
    flash_attention_fp8,
    flash_attention_fp8_cuda,
    flash_attention_fp8_pytorch,
    flash_attention_fp8_triton,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for custom kernels.")


def _assert_matches_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    expected = flash_attention_fp8_pytorch(q, k, v)
    triton_out = flash_attention_fp8_triton(q, k, v)
    cuda_out = flash_attention_fp8_cuda(q, k, v)

    assert torch.allclose(triton_out, expected, rtol=3e-2, atol=8e-2)
    assert torch.allclose(cuda_out, expected, rtol=3e-2, atol=8e-2)


@pytest.mark.parametrize("shape", [(1, 1, 16, 64), (2, 2, 33, 64), (1, 1, 65, 128)])
def test_flash_attention_fp8_forward_shapes(shape: tuple[int, ...]) -> None:
    torch.manual_seed(8)
    q = torch.randn(*shape, device="cuda", dtype=torch.float16)
    k = torch.randn(*shape, device="cuda", dtype=torch.float16)
    v = torch.randn(*shape, device="cuda", dtype=torch.float16)
    _assert_matches_reference(q, k, v)


def test_flash_attention_fp8_zero_rows_and_dispatch() -> None:
    q = torch.zeros((1, 1, 8, 64), device="cuda", dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    expected = flash_attention_fp8_pytorch(q, k, v)

    for backend in ("pytorch", "triton", "cuda"):
        out = flash_attention_fp8(q, k, v, backend=backend)
        assert torch.allclose(out, expected, rtol=3e-2, atol=8e-2)


def test_flash_attention_fp8_rejects_unsupported_inputs() -> None:
    q = torch.randn(1, 1, 8, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attention_fp8_pytorch(q, q, q)

    q32 = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="float16"):
        flash_attention_fp8_pytorch(q32, q32, q32)
