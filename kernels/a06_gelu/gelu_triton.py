import torch
import triton
import triton.language as tl

# Triton 3.x 的 tanh 在 libdevice 里（旧版的 tl.math.tanh 已移除）。
from triton.language.extra.cuda import libdevice


# ============================================================================
# Triton Kernel
# ============================================================================
# 每个 Triton program 处理 BLOCK_SIZE 个元素
# 类似 CUDA 里一个 block 处理一段连续数据
#
# 使用 tanh 近似 GELU，对应 PyTorch 的 F.gelu(x, approximate='tanh')：
#   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3) ) )
#
# 计算统一在 fp32 下进行，store 时由 Triton 自动转回输入 dtype。
# ============================================================================

@triton.jit
def _gelu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 program 的 id
    pid = tl.program_id(axis=0)

    # 当前 program 负责的元素下标范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 防止越界
    mask = offsets < n_elements

    # load 并升精度到 fp32 计算
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # tanh 近似 GELU
    kAlpha = 0.7978845608028654  # sqrt(2 / pi)
    kBeta = 0.044715
    inner = kAlpha * (x + kBeta * x * x * x)
    y = 0.5 * x * (1.0 + libdevice.tanh(inner))

    # store
    tl.store(y_ptr + offsets, y, mask=mask)


# ============================================================================
# Python Wrapper: return version
# ============================================================================
# Python 调用：
#   y = gelu_triton(x)
# ============================================================================

def gelu_triton(x: torch.Tensor, block_size: int = 1024):
    assert x.is_cuda
    assert x.is_contiguous()

    y = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (
        triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
    )
    _gelu_kernel[grid](
        x,
        y,
        n_elements,
        BLOCK_SIZE=block_size,
    )
    return y


def main():
    # 测试
    x = torch.randn(10000, device="cuda")
    y = gelu_triton(x)
    import torch.nn.functional as F
    ref = F.gelu(x, approximate="tanh")
    assert torch.allclose(y, ref, atol=1e-5)
    print("Test passed!")
