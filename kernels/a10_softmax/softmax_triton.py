import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 program 处理一行 (一个 token)
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    row = tl.load(
        input_ptr + pid * input_row_stride + col_offsets,
        mask=mask,
        other=-float("inf"),
    )
    # 上采样到 fp32 做归约，保证数值稳定 (safe softmax)
    row = row.to(tl.float32)
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = (numerator / denominator).to(output_ptr.dtype.element_ty)

    tl.store(
        output_ptr + pid * output_row_stride + col_offsets,
        softmax_out,
        mask=mask,
    )


def softmax_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.is_cuda
    assert x.dim() == 2
    assert x.is_contiguous()
    assert x.dtype in (torch.float32, torch.float16)

    n_rows, n_cols = x.shape
    assert n_cols > 0

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = min(max(BLOCK_SIZE // 256, 4), 16)
    num_stages = 2

    out = torch.empty_like(x)
    _softmax_kernel[(n_rows,)](
        out,
        x,
        x.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
