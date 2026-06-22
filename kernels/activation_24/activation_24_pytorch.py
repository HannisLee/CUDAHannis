import torch
import torch.nn.functional as F
import triton

from .activation_24_common import rank_keep_mask, validate_input


def activation_24_sparsity_pytorch(x: torch.Tensor) -> torch.Tensor:
    validate_input(x)
    if not x.is_contiguous():
        x = x.contiguous()

    last_dim = x.shape[-1]
    if last_dim == 0:
        return x.clone()

    padded_last_dim = triton.cdiv(last_dim, 4) * 4
    pad = padded_last_dim - last_dim
    padded = F.pad(x, (0, pad)) if pad else x

    grouped = padded.reshape(-1, padded_last_dim // 4, 4)
    offsets = torch.arange(padded_last_dim, device=x.device).reshape(1, -1, 4)
    valid = offsets < last_dim
    abs_values = torch.where(valid, grouped.abs(), torch.full_like(grouped, -float("inf")))
    keep = rank_keep_mask(abs_values, valid)
    sparse = torch.where(keep, grouped, torch.zeros_like(grouped))
    sparse = sparse.reshape(*x.shape[:-1], padded_last_dim)
    return sparse[..., :last_dim].contiguous()
