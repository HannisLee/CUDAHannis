import torch


def sum_pytorch(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    return torch.sum(x).reshape(1)
