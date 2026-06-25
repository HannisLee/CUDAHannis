# 使用 Nsight Systems 分析 kernel

这份文档用于分析本项目里的 PyTorch、Triton 和 CUDA extension vector add kernel。Nsight Systems 适合先看端到端时间线：Python 进程、CUDA API 调用、kernel launch、GPU kernel 执行区间、同步点和 CPU/GPU 是否有空洞。

## 1. 激活环境

```bash
source scripts/activate_env.sh
```

确认输出里能看到：

- `conda env: triton-cu118`
- `torch cuda version: 11.8`
- `GPU name: NVIDIA RTX A6000`
- `triton version: 3.3.1`

也可以检查 Nsight Systems：

```bash
which nsys
nsys --version
```

当前服务器 `shiva` 上检测到的 `nsys` 路径是：

```text
/home/han.li/.local/opt/nvidia/nsight-systems/2026.2.1/bin/nsys
```

## 2. 先跑通 benchmark

```bash
python -m eval.benchmarks.benchmark_vector_add
```

第一次运行 CUDA extension 或 Triton kernel 时会发生编译/JIT，时间线里会混入编译开销。正式 profiling 前建议先普通运行一次，让缓存生成完。

## 3. 采集 Nsight Systems 报告

建议把报告放到项目内 `profiles/`，不要放到共享 home 的隐藏全局缓存里。

```bash
mkdir -p profiles
nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --output=profiles/vector_add_nsys \
  python -m eval.benchmarks.benchmark_vector_add
```

常用参数说明：

- `--trace=cuda,nvtx,osrt`：记录 CUDA API、NVTX 标记和 OS runtime 事件。
- `--sample=none`：关闭 CPU 采样，减少学习阶段的噪声。
- `--cpuctxsw=none`：关闭上下文切换采集，报告更小。
- `--force-overwrite=true`：允许覆盖同名报告。
- `--output=profiles/vector_add_nsys`：生成 `profiles/vector_add_nsys.nsys-rep`。

## 4. 查看摘要

命令行先看统计：

```bash
nsys stats profiles/vector_add_nsys.nsys-rep
```

重点关注：

- CUDA Kernel Summary：哪些 kernel 被执行，耗时多少。
- CUDA API Summary：`cudaLaunchKernel`、同步、内存分配是否占比异常。
- GPU Kernel 时间是否稳定，是否有第一次 JIT 编译造成的异常长尾。

## 5. 打开 GUI 时间线

如果本机有图形界面或能转发 GUI：

```bash
nsys-ui profiles/vector_add_nsys.nsys-rep
```

没有 GUI 时，可以把 `.nsys-rep` 下载到本地安装了 Nsight Systems 的机器上打开。

时间线里建议依次看：

1. Python 主线程是否频繁同步 GPU。
2. CUDA API row 中 kernel launch 是否密集、是否有长时间空白。
3. GPU row 中 PyTorch、Triton、CUDA extension kernel 的执行时间。
4. 第一次运行是否包含 Triton JIT 或 PyTorch extension 编译开销；正式比较时应先 warmup。

## 6. 只分析某段代码

Nsight Systems 可以识别 NVTX 标记。后续可以在 Python benchmark 中加入：

```python
torch.cuda.nvtx.range_push("triton_add")
out = triton_add(x, y)
torch.cuda.nvtx.range_pop()
```

这样 GUI 时间线上会出现更清晰的范围标签。当前项目 benchmark 已经用 warmup/repeat 和 CUDA Event 做基础计时，Nsight Systems 用于观察 launch 和同步行为。

## 7. Nsight Systems 与 Nsight Compute 的分工

- Nsight Systems：先看全局时间线，判断瓶颈在 Python、kernel launch、同步、拷贝还是 GPU kernel。
- Nsight Compute：再深入单个 kernel，看 occupancy、memory throughput、warp stall、coalescing 等硬件指标。

推荐顺序是先 `nsys`，再对可疑 kernel 用 `ncu` 深入分析。

## 8. 多服务器共享 home 的注意事项

本项目通过 `scripts/activate_env.sh` 把缓存放到当前 hostname 独立目录：

- `TRITON_CACHE_DIR`
- `TORCH_EXTENSIONS_DIR`
- `CUDA_CACHE_PATH`

换服务器后先重新运行：

```bash
bash scripts/check_env.sh
source scripts/activate_env.sh
python -m eval.benchmarks.benchmark_vector_add
```

不要假设另一台服务器的 driver、CUDA toolkit、`nvcc`、`nsys` 或 `ncu` 版本和 `shiva` 相同。
