# Activation 2:4 Triton 算子优化记录

## 背景

这个算子的目标是对输入张量最后一维做 activation 2:4 sparsity：每连续 4 个元素作为一组，只保留绝对值最大的 2 个元素，其余位置写 0。遇到绝对值相等时，固定让更低下标优先保留，保证 PyTorch、Triton、CUDA extension 三个版本输出完全一致。

当前评测使用 `compare_single.py`，输入形状是 Qwen3.5-9B MLP intermediate activation 的近似真实尺寸：

```text
shape = (1, 1024, 12288)
```

其中 `12288` 是 MLP intermediate size。下面所有性能数据都只记录 Triton 版本，单位是毫秒。

## version1

### 总体思路

最初版本按“行 + group block”二维 grid 启动 Triton program。每一行的最后一维按 4 个元素一组，每个 program 处理若干个 4 元组。

### 如何实现

每个 Triton program 读取一批 group，每个 group 读取 `v0/v1/v2/v3` 四个元素。然后分别计算四个 lane 的 rank：

```text
rank = 组内比当前元素绝对值更大的元素数量
```

tie-break 通过 `>` 和 `>=` 的组合实现：如果绝对值相等，更低 lane 的元素排名更靠前。最后 `rank < 2` 的 lane 保留原值，其余写 0。

### 为什么这样做

这是最直接、最容易验证正确性的写法。它和 PyTorch reference 的逻辑高度一致，也能处理最后一维不是 4 的倍数的情况。为了支持不规则 shape，它在 load/store 时使用 mask，尾部越界元素被当成无效元素。

### 取舍

这个版本通用性好，但是对 Qwen3.5-9B 的真实 shape 来说有额外开销。`12288` 可以被 4 整除，也可以被很多 block 配置整除，因此通用 mask 和尾块逻辑在这个场景下是不必要的。

### Triton 数据

```text
float32: max error 0.00000000e+00, 0.1660 ms
float16: max error 0.00000000e+00, 0.0823 ms
```

## version2

### 改动方向

针对 Qwen3.5-9B 的 aligned shape 增加 fast path，减少通用路径里的 mask 和尾部处理开销。同时调小每个 program 处理的 group 数，改善并行粒度。

### 如何改动

新增 full-block fast path：当 `last_dim % 4 == 0` 且 group block 没有尾块时，进入无 mask kernel。这个路径中 load/store 都不再带 mask，也不再需要把无效元素置为 `-inf`。

同时默认 `block_groups` 从 256 调到 64，让每个 Triton program 处理更少的 group，从而增加 program 数，改善当前 shape 下的调度粒度。

### 为什么这样改

原始版本为了通用性保留了很多分支和 mask，但 Qwen3.5-9B 的 MLP intermediate size 是规整的。去掉 mask 可以减少 predicate 和条件处理，`block_groups=64` 在 sweep 中比 256 更稳。

### 实验和取舍

尝试过 pairwise top-2 selection，希望减少 rank 比较次数。但实际效果只小幅改善，说明瓶颈并不只是比较数量，也受到寄存器、访存和 launch 粒度影响。

### Triton 数据

```text
float32: max error 0.00000000e+00, 0.1624 ms
float16: max error 0.00000000e+00, 0.0776 ms
```

## version3

### 改动方向

从“按 4 元组分块”改为“按连续元素分块”的 fast path。核心目标是改善内存访问连续性，即使会引入一些重复读取。

### 如何改动

新增 contiguous-element fast path。每个 program 处理一段连续元素，先根据元素 offset 得到当前 lane：

```text
lane = offset % 4
group_base = offset - lane
```

然后读取当前元素所在 group 的四个元素，计算当前 lane 是否属于 top-2。这个路径让 store 是完全连续的，load 也围绕连续区域展开。

针对 dtype 做了 block size 调参：

```text
float32: BLOCK_SIZE = 128
float16: BLOCK_SIZE = 256
```

不规则 shape 仍保留旧的 masked/group kernel。

### 为什么这样改

version1/version2 的 group kernel 每个 program 内是按 `elem0/elem1/elem2/elem3` 四条向量分别访问，虽然逻辑清楚，但访存模式不如连续元素路径直接。对于大尺寸 `(1, 1024, 12288)`，连续读写能让内存访问更友好。

### 取舍

这个版本会重复读取同组邻居值：每个元素都会读取自己所在 group 的 4 个值。因此总 load 数量变多。但是它换来了更连续的访问模式和更好的实际性能。实测说明这个取舍在目标 shape 上是值得的。

### Triton 数据

```text
float32: max error 0.00000000e+00, 0.1465 ms
float16: max error 0.00000000e+00, 0.0756 ms
```

## version4

### 改动方向

在 version3 的 contiguous fast path 上减少冗余 load 和不必要的 rank 计算。

### 如何改动

version3 中当前元素 `v` 是单独 `tl.load(x + offsets)` 读出的，同时又读了 `v0/v1/v2/v3`。由于 `v` 必然等于这四个值中的一个，version4 改为：

```text
v = select(v0, v1, v2, v3) based on lane
```

这样可以去掉一次显式 self-value load。

另外，rank 计算从“四个 lane 都算 rank，再按 lane 选择一个 rank”改为“只计算当前 lane 的 rank”。tie-break 仍然按低 lane 优先。

### 为什么这样改

contiguous fast path 的主要开销之一是重复读取和重复比较。既然每个 program 中已经拿到了 group 的四个值，就应该复用这些值，而不是重新加载当前元素。

### 取舍

这个改动让逻辑更贴近“每个元素只判断自己是否保留”。但减少比较数量并不一定直接转化为大幅加速，因为新的 `tl.where` 选择和 lane-aware tie-break 也会占用指令和寄存器。最终收益较小。

### Triton 数据

```text
float32: max error 0.00000000e+00, 0.1463 ms
float16: max error 0.00000000e+00, 0.0756 ms
```

## version5

### 改动方向

继续围绕 contiguous fast path 优化地址计算和 launch 参数。

### 如何改动

首先把 lane 计算从全局 offset 改成局部 offset：

```text
local_offsets = tl.arange(0, BLOCK_SIZE)
offsets = pid * BLOCK_SIZE + local_offsets
lane = local_offsets & 3
```

由于 block 起点按 block size 对齐，而 block size 是 4 的倍数，所以局部 offset 的低 2 bit 就是 lane。这样可以简化地址相关依赖。

然后重新调参：

```text
float32: BLOCK_SIZE = 512, num_warps = 2
float16: BLOCK_SIZE = 256, num_warps = 4
```

### 为什么这样改

局部 lane 计算减少了对全局 offset 的依赖，表达式更简单。block size 和 warp 数的选择来自多轮 sweep：float32 更适合更大的 block 和较少 warps，float16 在 `256/4` 下更稳。

### 取舍

参数 sweep 的结果存在一定波动，因此只保留在正式 `compare_single.py` 口径下表现稳定的配置。部分 sweep 中看起来更快的组合，在正式脚本中不能稳定复现，因此没有采用。

### Triton 数据

```text
float32: max error 0.00000000e+00, 0.1459 ms
float16: max error 0.00000000e+00, 0.0748 ms
```

## version6

### 改动方向

尝试进一步减少 Python wrapper 侧和 fallback 准备逻辑的开销，同时评估更激进的 kernel 方向。

### 如何改动

aligned fast path 命中时，不再提前计算 fallback 才需要的 metadata：

```text
rows
groups_per_row
grid
```

这些值只在进入 fallback/full-group path 时才计算。这样 fast path 的 Python wrapper 更直接。

同时实验了两个方向：

1. 2D grouped-load kernel：试图一次按 `(groups, lanes)` 形状加载每个 4 元组，减少重复邻居 load。
2. cache modifier：给重复读取的邻居 load 加 `.ca`，试图改善缓存行为。

### 为什么这样改

version5 的 kernel 已经接近当前 CUDA extension 的耗时，明显的大结构优化空间变小。此时一个可行方向是减少 fast path 外围准备开销，另一个方向是探索是否能在 kernel 内减少重复 load。

### 实验结果和取舍

2D grouped-load 从理论上减少重复读取，但实测更慢。原因可能是 Triton 3.3.1 对 2D block 的代码生成、寄存器布局和 store 方式不如当前 1D contiguous path 高效。

cache modifier 对单次 fp16 有过收益，但不稳定，并且会让 fp32 变慢。为了避免引入不稳定复杂度，最终没有保留。

version6 的正式记录中，float16 与 version5 持平，float32 有波动退步。它说明 wrapper 层面的小改动不是主要瓶颈，真正瓶颈仍在 kernel 的访存和每元素判断逻辑上。

### Triton 数据

```text
float32: max error 0.00000000e+00, 0.1468 ms
float16: max error 0.00000000e+00, 0.0748 ms
```

## 总结

从 version1 到 version6，主要优化路径是：

1. 先建立正确且通用的 group-based kernel。
2. 针对 Qwen3.5-9B 的 aligned MLP intermediate shape 去掉 mask 和尾块逻辑。
3. 从 group-based 访问改为 contiguous-element 访问，利用更连续的读写模式换取更好的实际吞吐。
4. 在 contiguous path 上减少冗余 load、简化 lane 计算，并按 dtype 调整 block size 和 warp 数。
5. 对更激进的 2D grouped-load 和 cache modifier 做实验，但因为不稳定或更慢，没有保留。

当前最好的记录来自 version5：

```text
float32: 0.1459 ms
float16: 0.0748 ms
```

version6 保留了一些 wrapper 层面的整理，但它不是性能最优记录。后续如果继续优化，更可能需要从更底层的代码生成、向量化 store、或者专门的 CUDA kernel 角度入手，而不是继续微调当前 Triton 1D contiguous path。
