"""Activation 2:4 sparsity — Triton version1。

对应 OPTIMIZATION.md 的 version1：
- 按「行 + group block」二维 grid 启动，每个 program 处理 BLOCK_GROUPS 个 4 元组。
- load/store 全部带 mask，尾部越界 lane 当成无效元素（abs = -inf），
  因此能正确处理 last_dim 不是 4 的倍数的情况。
- tie-break：绝对值相等时更低 lane 优先，用 `>` 与 `>=` 的组合实现，
  保证与 PyTorch reference 完全一致。
- block_groups 默认 256（version2 才调到 64）。

这个版本通用性最好，但对 aligned shape 来说 mask / 尾块逻辑是多余开销。
本文件为独立 top-level 模块，不依赖包内相对导入。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _activation_24_sparsity_kernel(
    x_ptr,
    out_ptr,
    last_dim: tl.constexpr,
    groups_per_row: tl.constexpr,
    BLOCK_GROUPS: tl.constexpr,
):
    row = tl.program_id(axis=0)
    block = tl.program_id(axis=1)
    group_offsets = block * BLOCK_GROUPS + tl.arange(0, BLOCK_GROUPS)
    row_base = row * last_dim
    elem0 = group_offsets * 4
    elem1 = elem0 + 1
    elem2 = elem0 + 2
    elem3 = elem0 + 3

    group_mask = group_offsets < groups_per_row
    mask0 = group_mask & (elem0 < last_dim)
    mask1 = group_mask & (elem1 < last_dim)
    mask2 = group_mask & (elem2 < last_dim)
    mask3 = group_mask & (elem3 < last_dim)

    v0 = tl.load(x_ptr + row_base + elem0, mask=mask0, other=0.0)
    v1 = tl.load(x_ptr + row_base + elem1, mask=mask1, other=0.0)
    v2 = tl.load(x_ptr + row_base + elem2, mask=mask2, other=0.0)
    v3 = tl.load(x_ptr + row_base + elem3, mask=mask3, other=0.0)

    # 无效 lane 不参与 top-2：abs 置为 -inf
    a0 = tl.where(mask0, tl.abs(v0), -float("inf"))
    a1 = tl.where(mask1, tl.abs(v1), -float("inf"))
    a2 = tl.where(mask2, tl.abs(v2), -float("inf"))
    a3 = tl.where(mask3, tl.abs(v3), -float("inf"))

    # rank = 组内比当前 lane 绝对值更大的元素数量；
    # 相等时低 lane 优先，所以对方用 `>=` 当前、当前用 `>` 对方。
    rank0 = (a1 > a0).to(tl.int32) + (a2 > a0).to(tl.int32) + (a3 > a0).to(tl.int32)
    rank1 = (a0 >= a1).to(tl.int32) + (a2 > a1).to(tl.int32) + (a3 > a1).to(tl.int32)
    rank2 = (a0 >= a2).to(tl.int32) + (a1 >= a2).to(tl.int32) + (a3 > a2).to(tl.int32)
    rank3 = (a0 >= a3).to(tl.int32) + (a1 >= a3).to(tl.int32) + (a2 >= a3).to(tl.int32)

    tl.store(out_ptr + row_base + elem0, tl.where(rank0 < 2, v0, 0.0), mask=mask0)
    tl.store(out_ptr + row_base + elem1, tl.where(rank1 < 2, v1, 0.0), mask=mask1)
    tl.store(out_ptr + row_base + elem2, tl.where(rank2 < 2, v2, 0.0), mask=mask2)
    tl.store(out_ptr + row_base + elem3, tl.where(rank3 < 2, v3, 0.0), mask=mask3)


def activation_24_sparsity_triton(x: torch.Tensor, block_groups: int = 256) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()

    last_dim = x.shape[-1]
    out = torch.empty_like(x)
    if last_dim == 0:
        return out

    rows = x.numel() // last_dim
    groups_per_row = triton.cdiv(last_dim, 4)
    grid = (rows, triton.cdiv(groups_per_row, block_groups))

    _activation_24_sparsity_kernel[grid](
        x,
        out,
        last_dim,
        groups_per_row,
        BLOCK_GROUPS=block_groups,
    )
    return out


def main():
    import time
    import torch

    torch.manual_seed(0)
    device = "cuda"

    # 单个测试 tensor：last_dim % 4 == 0，但 v1 始终走 masked group kernel 路径
    shape = (4096, 4096)
    dtype = torch.float16

    x = torch.randn(shape, device=device, dtype=dtype)

    # 预热：触发 CUDA 初始化 + Triton JIT 编译
    for _ in range(10):
        y = activation_24_sparsity_triton(x)

    torch.cuda.synchronize()

    # 正式性能测试
    iters = 100
    start = time.perf_counter()

    for _ in range(iters):
        y = activation_24_sparsity_triton(x)

    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / iters

    print(f"shape: {shape}")
    print(f"dtype: {dtype}")
    print(f"avg latency: {avg_ms:.4f} ms")

    # NCU 精准采集区：只采这一发 kernel
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()

    for _ in range(1000):
        y = activation_24_sparsity_triton(x)

    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print("NCU capture kernel finished.")


if __name__ == "__main__":
    main()
