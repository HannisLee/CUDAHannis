import torch
import triton
import triton.language as tl


# ============================================================================
# Triton Kernel
# ============================================================================
# 每个 Triton program 处理 BLOCK_SIZE 个元素
# 类似 CUDA 里一个 block 处理一段连续数据
# ============================================================================

@triton.jit
def _vector_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 program 的 id
    pid = tl.program_id(axis=0)

    # 当前 program 负责的元素下标范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 防止越界
    mask = offsets < n_elements

    # load
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)

    # compute
    c = a + b

    # store
    tl.store(c_ptr + offsets, c, mask=mask)


# ============================================================================
# Python Wrapper: return version
# ============================================================================
# Python 调用：
#   c = vector_add_triton(a, b)
# ============================================================================

def vector_add_triton(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024):
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    assert a.is_contiguous()
    assert b.is_contiguous()
    c = torch.empty_like(a)
    n_elements = a.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )
    _vector_add_kernel[grid](
        a,
        b,
        c,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return c

def main():
    # 测试
    a = torch.randn(10000, device="cuda")
    b = torch.randn(10000, device="cuda")
    c = vector_add_triton(a, b)
    assert torch.allclose(c, a + b)
    print("Test passed!")