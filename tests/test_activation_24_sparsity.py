import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.activation_24 import (
    activation_24_sparsity_cuda,
    activation_24_sparsity_pytorch,
    activation_24_sparsity_triton,
    sparsify_before_up_gate,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请检查 nvidia-smi、driver、PyTorch CUDA wheel 和环境激活状态。")


def _assert_matches_reference(x: torch.Tensor) -> None:
    expected = activation_24_sparsity_pytorch(x)
    triton_out = activation_24_sparsity_triton(x)
    cuda_out = activation_24_sparsity_cuda(x)

    assert torch.equal(triton_out, expected)
    assert torch.equal(cuda_out, expected)


def test_activation_24_sparsity_float32_and_float16() -> None:
    _require_cuda()
    torch.manual_seed(24)

    for dtype in (torch.float32, torch.float16):
        for shape in ((2, 16), (3, 17), (2, 5, 33)):
            x = torch.randn(*shape, device="cuda", dtype=dtype)
            _assert_matches_reference(x)


def test_activation_24_sparsity_tie_break_and_padding() -> None:
    _require_cuda()

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
    _require_cuda()
    x = torch.randn(4, 19, device="cuda", dtype=torch.float32)
    expected = activation_24_sparsity_pytorch(x)

    for backend in ("pytorch", "triton", "cuda"):
        assert torch.equal(sparsify_before_up_gate(x, backend=backend), expected)


def main() -> None:
    _require_cuda()
    for dtype in (torch.float32, torch.float16):
        x = torch.randn(4, 16, 4099, device="cuda", dtype=dtype)
        expected = activation_24_sparsity_pytorch(x)
        triton_out = activation_24_sparsity_triton(x)
        cuda_out = activation_24_sparsity_cuda(x)
        triton_error = (triton_out - expected).abs().max().item()
        cuda_error = (cuda_out - expected).abs().max().item()
        print(f"dtype: {dtype}")
        print(f"triton max error: {triton_error:.8e}")
        print(f"cuda max error: {cuda_error:.8e}")
        assert torch.equal(triton_out, expected)
        assert torch.equal(cuda_out, expected)
    print("Activation 2:4 sparsity correctness: PASS")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "CUDA_HOME" in str(exc) or "nvcc" in str(exc):
            pytest.skip(str(exc))
        raise
