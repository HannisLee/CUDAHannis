import pytest
import torch

from kernels.activation_24 import (
    activation_24_sparsity_cuda,
    activation_24_sparsity_pytorch,
    activation_24_sparsity_triton,
    sparsify_before_up_gate,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for custom kernels.")


def _assert_matches_reference(x: torch.Tensor) -> None:
    expected = activation_24_sparsity_pytorch(x)
    triton_out = activation_24_sparsity_triton(x)
    cuda_out = activation_24_sparsity_cuda(x)

    assert torch.equal(triton_out, expected)
    assert torch.equal(cuda_out, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(2, 16), (3, 17), (2, 5, 33)])
def test_activation_24_sparsity_float32_and_float16(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    torch.manual_seed(24)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    _assert_matches_reference(x)


def test_activation_24_sparsity_tie_break_and_padding() -> None:
    x = torch.tensor(
        [
            [2.0, -2.0, 2.0, -2.0, 1.0, 1.0, 1.0],
            [0.1, -0.2, 0.3, -0.4, -3.0, 2.0, -1.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected = torch.tensor(
        [
            [2.0, -2.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.3, -0.4, -3.0, 2.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    assert torch.equal(activation_24_sparsity_pytorch(x), expected)
    assert torch.equal(activation_24_sparsity_triton(x), expected)
    assert torch.equal(activation_24_sparsity_cuda(x), expected)


def test_sparsify_before_up_gate_backends() -> None:
    x = torch.randn(4, 19, device="cuda", dtype=torch.float32)
    expected = activation_24_sparsity_pytorch(x)

    for backend in ("pytorch", "triton", "cuda"):
        assert torch.equal(sparsify_before_up_gate(x, backend=backend), expected)
