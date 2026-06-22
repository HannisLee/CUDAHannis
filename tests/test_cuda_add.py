import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kernels.vector_add import cuda_add


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用：请检查 nvidia-smi、driver、PyTorch CUDA wheel 和环境激活状态。")

    torch.manual_seed(1)
    device = torch.device("cuda")
    x = torch.randn(1_000_003, device=device, dtype=torch.float32)
    y = torch.randn(1_000_003, device=device, dtype=torch.float32)

    out = cuda_add(x, y)
    expected = x + y
    max_error = (out - expected).abs().max().item()
    print(f"max error: {max_error:.8e}")

    assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)
    print("CUDA vector add correctness: PASS")


def test_cuda_add() -> None:
    try:
        main()
    except RuntimeError as exc:
        if "CUDA_HOME" in str(exc) or "nvcc" in str(exc):
            pytest.skip(str(exc))
        raise


if __name__ == "__main__":
    main()
