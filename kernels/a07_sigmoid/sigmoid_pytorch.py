import torch


def sigmoid_pytorch(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    return torch.sigmoid(x)
