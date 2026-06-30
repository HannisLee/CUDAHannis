import torch


def max_pytorch(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    return torch.max(x).reshape(1)
