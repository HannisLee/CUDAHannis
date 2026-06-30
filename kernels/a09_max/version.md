# a09_max 各版本 `max` 归约算子实现与优化思路

> 本目录实现了一个 **1-D `max` 归约算子**（求张量所有元素的最大值），共 **6 个 CUDA 版本**：`v0`、`v1`、`v2`、`v4`、`v5`、`v6`（**注意：没有 `v3`**，作者跳过了这一号），并附 PyTorch、Triton 两个参考实现用于对照。
> 全部版本输入输出均为 **fp32、连续（contiguous）的 CUDA 张量**；所有版本的输出与 PyTorch 参考值 `max_abs = 0.000e+00`（`max` 归约无浮点误差，结果位级精确）。

---

## 一、优化主线一览

从最慢到最快，每一版都对应**打破上一个版本的一个明确瓶颈**：

| 版本 | 核心手段 | 相对 v0 加速比 (K=4096) | 解决的瓶颈 |
|------|----------|:---:|------|
| **v0** | 单线程串行 for 循环 | 1× | — (基线，演示"为何要并行") |
| **v1** | 共享内存树形归约 + 两阶段 | ~1340× | 串行 → block 级并行 |
| **v2** | `__shfl_down_sync` Warp Shuffle | ~2090× | 共享内存延迟 / bank conflict / `__syncthreads` |
| **v4** | `float4` 向量化加载（每线程 4 元素） | ~3830× | 内存带宽未用满、访存指令过多 |
| **v5** | 每线程 8 元素 + **单 kernel**（atomic last-block） | ~4260× | 消除第二次 kernel launch 的开销与同步 |
| **v6** | **递归多趟**归约 + 双跨步加载 + `shfl_xor` | ~4580× | 超大 N 时单 block 收尾的并行度不足 |

---

## 二、公共基础设施

在进入各版本之前，先说明几个被反复复用的通用构件。

### 2.1 两阶段归约（two-stage reduction）

除 v0（完全串行）和 v5/v6（自带收尾逻辑）外，`v1 / v2 / v4` 都采用**"stage kernel + finalize kernel"两阶段**结构：

1. **stage kernel**：每个 block 归约自己负责的一段数据，得到 1 个**部分最大值（partial）**写入 `partial[blockIdx.x]`；
2. **finalize kernel**：再用 1 个 block 把所有 partial 归约成最终结果写入 `y[0]`。

原因：单个 block 内的归约只能借助 shared memory / warp shuffle，跨 block 必须经过 global memory 中转，所以需要"先各自出 partial，再汇总"。

### 2.2 `max_finalize_kernel`（共享收尾核函数）

`v1 / v2 / v4` 共用同一个 finalize 核函数 `max_finalize_kernel`：

```cpp
for (int i = tid; i < n_partials; i += blockDim.x)   // grid-stride：一个 block 扛下所有 partial
    val = fmaxf(val, partial[i]);
// 然后在 block 内用共享内存树形归约得到最终值
```

它用 **grid-stride 循环**让单个 256 线程的 block 跑完所有 partial，再做一次块内树形归约。这种"一个 block 收尾"的写法在 partial 数量不大时没问题，但在 N 极大、partial 数极多时会成为瓶颈——这正是 v5/v6 要优化的地方。

### 2.3 `FLOAT4` 向量化加载宏

```cpp
#define FLOAT4(value) (reinterpret_cast<const float4 *>(&(value))[0])
```

把连续 4 个 float（16 字节）一次性作为一条 128-bit `float4` 加载指令读入寄存器，减少访存指令数、提高带宽利用率，也利于对齐合并访问。

### 2.4 Warp Shuffle 归约原语

```cpp
#pragma unroll
for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xFFFFFFFF, val, offset);
    val = fmaxf(val, other);
}
```

这是归约问题的"黄金片段"：warp 内无需 shared memory、无需 `__syncthreads`，5 步（offset = 16/8/4/2/1）就能把一个 warp（32 lane）归约到 **lane 0**。`__shfl_down_sync` 是"下移"，只让 lane 0 拿到结果；`v6` 用的是 `__shfl_xor_sync`（蝶形/全归约），见 §3.6。

---

## 三、各版本详解

### 3.1 v0 —— 单线程串行（基线）

```cpp
__global__ void max_v0_kernel(const float *x, float *y, int N) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {   // 只用 1 个线程
    float max_val = -FLT_MAX;
    for (int i = 0; i < N; ++i)
      max_val = fmaxf(max_val, x[i]);
    y[0] = max_val;
  }
}
// launch: <<<1, 1>>>
```

- **Launch 配置**：`<<<1, 1>>>`，整个 GPU 上只有 **1 个线程**串行遍历全部 N 个元素。
- **目的**：纯粹的正确性基线，用来对照"不并行有多慢"，并验证后续版本的数值正确性。
- **性能**：K=4096 时 **351 ms**，K=8192 时 **703 ms**；时间随 N **线性翻倍**（K 翻倍时间翻倍），完美符合 O(N) 串行特征。完全不可用于生产。

---

### 3.2 v1 —— 共享内存树形归约

```cpp
__shared__ float smem[BLOCK_SIZE];
smem[tid] = (idx < N) ? x[idx] : -FLT_MAX;     // 每线程 1 元素
__syncthreads();
for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {  // 树形折半
  if (tid < offset)
    smem[tid] = fmaxf(smem[tid], smem[tid + offset]);
  __syncthreads();
}
partial[blockIdx.x] = smem[0];                  // 块内最大值 → partial
```
然后接 §2.2 的 `max_finalize_kernel`。

- **每线程处理 1 个元素**；block 内用经典的"折半树形归约"，活跃线程保持连续（`tid < offset`），这是 Mark Harris 归约教程里的**较优变体**（避免跨 warp 发散、bank conflict 较少）。
- **相对 v0 的突破**：把串行变成数千个 block 的并行，K=4096 时 **0.262 ms**，比 v0 快 ~1340×。
- **残留瓶颈**：① 每层归约都要 `__syncthreads`（共享内存同步开销）；② 共享内存读写有延迟、可能有 bank conflict；③ **每线程只处理 1 个元素**，严重浪费带宽和 ILP。

---

### 3.3 v2 —— Warp Shuffle 归约

把 v1 块内共享内存树形归约**换成 warp shuffle**（§2.4 的黄金片段），只保留一个极小的 `warp_max_s[32]` 共享数组用于 warp 间交换：

```cpp
// 1) 每个 warp 内 shuffle 归约 → lane 0 得到该 warp 最大值
... __shfl_down_sync(0xFFFFFFFF, val, offset) ...
if (lane_id == 0) warp_max_s[warp_id] = val;
__syncthreads();
// 2) 第 0 个 warp 再对 8 个 warp 的结果做一次 shuffle 归约
```

- **相对 v1 的突破**：warp 内归约完全在**寄存器**里完成，不再每层读写 shared memory，也省掉了大量 `__syncthreads`；`#pragma unroll` 后编译器可激进优化。
- K=4096 时 **0.168 ms**，比 v1 再快约 1.56×。
- **残留瓶颈**：仍然**每线程 1 个元素**，访存带宽依然是软肋。

---

### 3.4 v4 —— `float4` 向量化加载

```cpp
int idx = (blockDim.x * blockIdx.x + tid) * 4;   // 每线程跨度 4
float4 reg_x = FLOAT4(x[idx]);                    // 一条 128-bit 指令读 4 个 float
val = fmaxf(fmaxf(reg_x.x, reg_x.y), fmaxf(reg_x.z, reg_x.w));  // 先线程内归约 4 个
// 再走 v2 的 warp shuffle 归约
```
- **Grid**：`(N + BLOCK_SIZE*4 - 1) / (BLOCK_SIZE*4)`，即每个 block 1024 元素，block 数量比 v2 少 4×。
- **相对 v2 的突破**：① **数据并行（thread coarsening）**——每线程先在自己的寄存器里把 4 个元素归约成 1 个，天然带来指令级并行（ILP）；② **向量化加载**把访存指令数压到 1/4，带宽利用率大增；③ block 数减少 → partial 数减少 → finalize 更轻。
- K=4096 时 **0.0917 ms**，比 v2 快约 1.83×。注意这里"先线程内、再线程间"的**两段式归约**是后续 v5/v6 都沿用的关键思路。

---

### 3.5 v5 —— 每线程 8 元素 + 单 kernel（atomic last-block 收尾）

这是**唯一一个把两阶段合并成单次 kernel launch** 的版本（不再调用 finalize kernel）。

**每线程 8 元素**（循环步长 4，即 2 次 `float4`）：
```cpp
int idx = (blockDim.x * blockIdx.x + tid) * V5_ITEMS_PER_THREAD;  // = *8
#pragma unroll
for (int i = 0; i < 8; i += 4) {                 // 读 2 个 float4
    float4 reg_x = FLOAT4(x[idx + i]);
    val = fmaxf(val, max(reg_x.x, reg_x.y, reg_x.z, reg_x.w));
}
// warp shuffle → warp_max_s → block 归约（同前）
```

**核心：last-block-does-finalize 技巧**（grid 级同步却不需要第二次 launch）：
```cpp
if (tid == 0) {
    partial[blockIdx.x] = val;        // 写自己的 partial
    __threadfence();                   // 保证 partial 对其他 block 可见 + 有序
    int ticket = atomicAdd(counter, 1);
    is_last_block = (ticket == gridDim.x - 1);  // 拿到最后一张票 = 我是最后一个
}
__syncthreads();

if (is_last_block) {                   // 只有最后完成的那个 block 做最终归约
    // grid-stride 扫描 partial[0 .. gridDim.x)，再块内 warp shuffle
    y[0] = final_val;
    *counter = 0;                       // 复位计数器，使 kernel 可重入/可重复调用
}
```

- **相对 v4 的突破**：① **数据并行再加码**（4 → 8 元素/线程）；② **省掉第二次 kernel launch**——避免了 launch 开销和显式的跨 kernel 同步；③ 用 `atomicAdd` + `__threadfence` 实现"最后到达的 block 负责收尾"，是经典的 grid 级协作归约模式。
- K=4096 时 **0.0825 ms**，比 v4 再快约 1.11×。
- **残留瓶颈**：收尾只由**一个 block** 完成，当 N 极大、grid 很大时，这个 block 要用 grid-stride 扫描成千上万个 partial，并行度受限——这正是 v6 要解决的。

> 注意：v5 用 `cudaMemsetAsync` 把 `counter` 清零，并在 kernel 结尾 `*counter = 0` 复位，因此**可被重复调用**。

---

### 3.6 v6 —— 递归多趟归约 + 双跨步加载 + `shfl_xor`

不再用"一个 block 收尾"，而是在 **host 端用 while 循环递归地多趟归约**，每趟都用满 block 并行度：

**Host 侧递归**：
```cpp
while (current_n > 1) {
    int n_blocks = (current_n + V6_ELEMENTS_PER_BLOCK - 1) / V6_ELEMENTS_PER_BLOCK;
    out_ptr = (n_blocks == 1) ? y : new_partial;   // 最后一趟直接写 y
    max_v6_stage_kernel<<<n_blocks, block>>>(current_ptr, out_ptr, current_n);
    current_ptr = out_ptr;
    current_n = n_blocks;                          // 每趟把 N 缩到 ~1/1024
}
// 特判 current_n == 1 → cudaMemcpyAsync 直接拷贝
```

**双跨步加载（ILP 关键）**：每线程读**两个**相隔半 block 的 `float4`，让两条访存指令相互独立、便于内存系统流水线化：
```cpp
int idx0 = block_base + tid * 4;                       // 前 512 元素段
int idx1 = block_base + (V6_ELEMENTS_PER_BLOCK/2) + tid * 4;  // 后 512 元素段
// 各自 FLOAT4 加载并归约进同一个 val
```

**`__shfl_xor_sync` 蝶形归约**（区别于 v2/v4/v5 的 `__shfl_down_sync`）：
```cpp
for (int mask = WARP_SIZE >> 1; mask > 0; mask >>= 1) {
    float other = __shfl_xor_sync(0xFFFFFFFF, val, mask);  // lane ↔ lane^mask
    val = fmaxf(val, other);
}
```
`shfl_xor` 是蝶形 all-reduce：循环结束后**每个 lane** 都持有完整归约结果（对 `max` 这类满足交换结合律的运算成立）。本版本最终只取 lane 0，所以与 `shfl_down` 结果一致，使用 `xor` 更多是风格/可扩展选择（便于后续广播）。

- **配置**：block 128、每线程 8 元素 → 每 block **1024 元素**。
- **相对 v5 的突破**：收尾不再靠"一个 block 扫所有 partial"，而是**递归地每一趟都用全部 block 并行**——趟数随 N 增长极慢（N 每大 1024× 才多一趟），因此**对超大 N 友好**。
- **取舍**：多趟 = 多次 kernel launch，**小 N 时 launch 开销反而拖累**（见 K=512 时 v5 略快于 v6）；大 N 时优势显现。
- K=8192（33.5M 元素）时 **0.147 ms**，**反超 PyTorch（0.162 ms）**，与 Triton 持平。

---

## 四、优化思路总结（脉络图）

```
v0  串行单线程
 │   瓶颈: 完全没有并行
 ▼
v1  共享内存树形归约 + 两阶段          ← 引入 block 级并行
 │   瓶颈: smem 同步延迟、每线程 1 元素
 ▼
v2  Warp Shuffle                      ← 归约搬进寄存器，去 smem/去 syncthreads
 │   瓶颈: 仍每线程 1 元素, 带宽浪费
 ▼
v4  float4 向量化 + 线程内归约 4 元素   ← 数据并行(ILP) + 带宽 4×
 │   瓶颈: 仍有第二次 kernel launch
 ▼
v5  每线程 8 元素 + 单 kernel          ← atomic last-block 省掉第二次 launch
 │   瓶颈: 收尾单 block, 超大 N 并行度不足
 ▼
v6  递归多趟 + 双跨步 + shfl_xor        ← 每趟全并行收尾, 适配超大 N
```

可归纳为四条主线优化：
1. **并行度**：串行 → block 并行 → 递归全并行收尾。
2. **归约介质**：共享内存 → 寄存器（warp shuffle）。
3. **数据并行 / ILP**：每线程 1 → 4（`float4`）→ 8 元素，配以"线程内先归约、线程间再归约"的两段式。
4. **kernel 调度**：两次 launch → 单 kernel（atomic 收尾）→ 递归多趟（按需自适应）。

---

## 五、性能对比（来自 `results.txt`，N=4096，fp32，单位 ms）

| K | 元素数 | torch | v0 | v1 | v2 | v4 | v5 | v6 | triton |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 512 | 2.1M | 0.0158 | 45.19 | 0.0350 | 0.0222 | 0.0145 | **0.0136** | 0.0154 | 0.0550 |
| 1024 | 4.2M | 0.0240 | 87.90 | 0.0661 | 0.0422 | 0.0253 | **0.0234** | 0.0241 | 0.0553 |
| 2048 | 8.4M | 0.0415 | 175.79 | 0.1314 | 0.0842 | 0.0473 | 0.0432 | **0.0416** | 0.0557 |
| 4096 | 16.8M | 0.0766 | 351.63 | 0.2621 | 0.1683 | 0.0917 | 0.0825 | **0.0768** | 0.0767 |
| 8192 | 33.6M | 0.1623 | 703.13 | 0.5233 | 0.3360 | 0.1797 | 0.1616 | **0.1470** | 0.1471 |

**观察**：
- **v0** 是反面教材：时间随 N 线性增长，K=8192 时高达 703 ms，比最优版慢近 4800×。
- **每一版相比上一版都有明显提升**，优化路径连贯有效。
- **v5 在中小 N（K ≤ 4096）最快**：单 kernel 无 launch 开销、数据并行充分。
- **v6 在超大 N（K=8192）最快，并反超 PyTorch**：递归多趟让收尾阶段也保持满并行，弥补了 launch 次数多的代价。
- 自 v4 起，自研 CUDA 版本全面**超越 Triton 参考实现**；v6 在大 N 下**与 PyTorch 官方 cudnn/cub 实现持平甚至更优**。

> 结论：**v5 与 v6 是两个最终候选**——中小规模选 v5（少 launch），超大 N 选 v6（收尾并行）。

---

## 六、关键技术索引

| 技术 | 出现版本 | 作用 |
|------|----------|------|
| 两阶段归约（stage + finalize） | v1 / v2 / v4 | 跨 block 必须借 global memory 中转 |
| 共享内存树形归约（折半） | v1, finalize | 块内归约的经典写法 |
| `__shfl_down_sync` warp 归约 | v2 / v4 / v5 | 归约搬进寄存器，去 smem/去 syncthreads |
| `float4` 向量化加载（`FLOAT4`） | v4 / v5 / v6 | 128-bit 合并访存，带宽 4× |
| 线程内先归约（thread coarsening / ILP） | v4 / v5 / v6 | 每线程多元素，提升 ILP 与算访比 |
| 每线程 8 元素 | v5 / v6 | 进一步榨取数据并行 |
| **atomic last-block finalize** | v5 | 单 kernel 完成 grid 级归约，省第二次 launch |
| `__threadfence` | v5 | 保证 partial 写入对其他 block 可见且有序 |
| `cudaMemsetAsync` + `*counter=0` 复位 | v5 | 使 atomic 计数器版本可重入 |
| **递归多趟归约**（host while 循环） | v6 | 收尾阶段也保持满 block 并行，适配超大 N |
| 双跨步加载（idx0 / idx1 隔半 block） | v6 | 两条独立访存指令，提升内存流水线 ILP |
| `__shfl_xor_sync` 蝶形 all-reduce | v6 | 每个 lane 都拿到完整结果，便于广播 |
| `cudaMemcpyAsync` 边界特判 | v6 | 处理 N=1 等退化情形 |

---

## 附：文件说明

| 文件 | 说明 |
|------|------|
| `max.cu` | 全部 6 个 CUDA 版本的 kernel 与 PyTorch 绑定（`max_v0` ~ `max_v6`） |
| `benchmark.py` | JIT 编译 `max.cu`，对每种 shape 跑 warmup+iters 计时并与参考对比 |
| `max_pytorch.py` | PyTorch 参考实现 `torch.max` |
| `max_triton.py` | Triton 参考实现（block_size=1024，递归多趟，思路与 v6 类似） |
| `results.txt` | benchmark 的输出（即本文 §五 数据来源） |
