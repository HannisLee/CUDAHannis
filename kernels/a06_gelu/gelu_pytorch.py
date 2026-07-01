import torch
import torch.nn.functional as F


def gelu_pytorch(x: torch.Tensor) -> torch.Tensor:
    # 使用 tanh 近似，与 CUDA/Triton kernel 的实现保持一致
    if not x.is_contiguous():
        x = x.contiguous()
    return F.gelu(x, approximate="tanh")
