import pytest
import torch

from kernels.vector_add import cuda_add, triton_add


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for custom kernels.")


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1,), (3, 17), (2, 5, 257)])
def test_vector_add_matches_pytorch(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    torch.manual_seed(1)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    y = torch.randn(*shape, device="cuda", dtype=dtype)
    expected = x + y
    tolerance = 1e-3 if dtype == torch.float16 else 1e-6

    assert torch.allclose(triton_add(x, y), expected, rtol=tolerance, atol=tolerance)
    assert torch.allclose(cuda_add(x, y), expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("backend", [triton_add, cuda_add])
def test_vector_add_rejects_incompatible_inputs(backend) -> None:
    x = torch.randn(2, 4, device="cuda")
    with pytest.raises(ValueError, match="same shape"):
        backend(x, torch.randn(2, 5, device="cuda"))
    with pytest.raises(ValueError, match="same dtype"):
        backend(x, torch.randn(2, 4, device="cuda", dtype=torch.float16))

