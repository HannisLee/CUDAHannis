import torch


def activation_24_sparsity(x: torch.Tensor) -> torch.Tensor:
   
    last_dim = x.shape[-1]
    pad = (4 - last_dim % 4) % 4

    if pad > 0:
        x_pad = torch.nn.functional.pad(x, (0, pad))
    else:
        x_pad = x

    new_shape = x_pad.shape[:-1] + (-1, 4)
    x_grouped = x_pad.view(new_shape)

    abs_grouped = x_grouped.abs()

    # 每组取 top-2
    _, indices = torch.topk(abs_grouped, k=2, dim=-1)

    mask = torch.zeros_like(x_grouped, dtype=torch.bool)
    mask.scatter_(-1, indices, True)

    out = torch.where(mask, x_grouped, torch.zeros_like(x_grouped))

    out = out.view(x_pad.shape)

    if pad > 0:
        out = out[..., :last_dim]

    return out