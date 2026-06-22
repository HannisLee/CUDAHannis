import torch


def validate_input(x: torch.Tensor) -> None:
    if not x.is_cuda:
        raise RuntimeError("activation 2:4 sparsity expects a CUDA tensor.")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError(f"activation 2:4 sparsity supports float16/float32, got {x.dtype}.")
    if x.dim() == 0:
        raise ValueError("activation 2:4 sparsity expects at least 1 dimension.")


def rank_keep_mask(abs_values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(4, device=abs_values.device)
    greater = abs_values.unsqueeze(-2) > abs_values.unsqueeze(-1)
    equal_lower = (abs_values.unsqueeze(-2) == abs_values.unsqueeze(-1)) & (
        positions.view(1, 1, 4) < positions.view(1, 1, 4, 1)
    )
    rank = (greater | equal_lower).sum(dim=-1)
    return (rank < 2) & valid
