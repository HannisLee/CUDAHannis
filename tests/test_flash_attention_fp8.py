import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.flash_attention_fp8 import (
    flash_attention_fp8,
    flash_attention_fp8_cuda,
    flash_attention_fp8_pytorch,
    flash_attention_fp8_triton,
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请检查 nvidia-smi、driver、PyTorch CUDA wheel 和环境激活状态。")


def _assert_matches_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    expected = flash_attention_fp8_pytorch(q, k, v)
    triton_out = flash_attention_fp8_triton(q, k, v)
    cuda_out = flash_attention_fp8_cuda(q, k, v)

    assert torch.allclose(triton_out, expected, rtol=3e-2, atol=8e-2)
    assert torch.allclose(cuda_out, expected, rtol=3e-2, atol=8e-2)


def test_flash_attention_fp8_forward_shapes() -> None:
    _require_cuda()
    torch.manual_seed(8)
    for shape in ((1, 1, 16, 64), (1, 2, 33, 64), (1, 1, 65, 128)):
        q = torch.randn(*shape, device="cuda", dtype=torch.float16)
        k = torch.randn(*shape, device="cuda", dtype=torch.float16)
        v = torch.randn(*shape, device="cuda", dtype=torch.float16)
        _assert_matches_reference(q, k, v)


def test_flash_attention_fp8_zero_rows_and_dispatch() -> None:
    _require_cuda()
    q = torch.zeros((1, 1, 8, 64), device="cuda", dtype=torch.float16)
    k = torch.zeros_like(q)
    v = torch.randn_like(q)
    expected = flash_attention_fp8_pytorch(q, k, v)

    for backend in ("pytorch", "triton", "cuda"):
        out = flash_attention_fp8(q, k, v, backend=backend)
        assert torch.allclose(out, expected, rtol=3e-2, atol=8e-2)


def test_flash_attention_fp8_rejects_unsupported_inputs() -> None:
    _require_cuda()
    q = torch.randn(1, 1, 8, 32, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="head_dim"):
        flash_attention_fp8_pytorch(q, q, q)

    q32 = torch.randn(1, 1, 8, 64, device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="float16"):
        flash_attention_fp8_pytorch(q32, q32, q32)


def main() -> None:
    _require_cuda()
    torch.manual_seed(8)
    for shape in ((1, 1, 16, 64), (1, 2, 33, 64), (1, 1, 65, 128)):
        q = torch.randn(*shape, device="cuda", dtype=torch.float16)
        k = torch.randn(*shape, device="cuda", dtype=torch.float16)
        v = torch.randn(*shape, device="cuda", dtype=torch.float16)
        expected = flash_attention_fp8_pytorch(q, k, v)
        triton_out = flash_attention_fp8_triton(q, k, v)
        cuda_out = flash_attention_fp8_cuda(q, k, v)
        print(f"shape: {shape}")
        print(f"triton max error: {(triton_out - expected).abs().max().item():.8e}")
        print(f"cuda max error: {(cuda_out - expected).abs().max().item():.8e}")
        assert torch.allclose(triton_out, expected, rtol=3e-2, atol=8e-2)
        assert torch.allclose(cuda_out, expected, rtol=3e-2, atol=8e-2)
    print("FP8 causal FlashAttention forward correctness: PASS")


if __name__ == "__main__":
    main()
