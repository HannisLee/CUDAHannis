# CUDA / Triton 学习项目

这个项目用于从零学习 CUDA、Triton、PyTorch GPU 编程和自定义算子开发。当前项目提供 vector add、LLM MLP 前激活 2:4 稀疏、NVFP4-style 量化、FP8 causal FlashAttention forward 四类示例算子，并配套环境检查、激活脚本、正确性测试和简单 benchmark。

## 当前环境

- 服务器：`shiva`
- conda 环境：`triton-cu118`
- PyTorch CUDA runtime：`cu118`
- Triton 版本：由 PyTorch 依赖固定，目前为 `3.3.1`
- CUDA extension 编译依赖：`cuda-nvcc/cuda-cudart/cuda-cudart-dev/cuda-cccl 11.8.89`，安装在 `triton-cu118` 环境内
- CUDA extension host compiler：`gcc_linux-64/gxx_linux-64 11.4.0`

选择 `cu118` 的原因：当前服务器 driver 为 `555.42.06`，`nvidia-smi` 显示 CUDA Version `12.5`。这个值表示 driver 支持的最高 CUDA runtime 能力，不等于系统安装的 CUDA toolkit 版本。虽然 driver 可运行 CUDA 12.x 程序，但 Triton 会 JIT 编译 GPU 代码，优先选择 PyTorch 官方稳定提供的 `cu118` wheel 更保守、兼容性更好。

当前机器在 base 环境 PATH 中能看到 `nvcc 12.9` 和 Nsight Compute，但激活 `triton-cu118` 后不会强行把这些工具塞回 PATH。原因是 home 目录跨服务器共享，`nvcc`、Nsight 和 driver 版本可能不匹配；需要使用 CUDA toolkit 或 profiling 工具时，先运行 `bash scripts/check_env.sh` 确认当前服务器实际可用的工具路径和版本。

为了编译本项目的 CUDA extension，`triton-cu118` 环境内单独安装了 CUDA 11.8 的 `nvcc`、runtime headers 和 GCC/G++ 11。`scripts/activate_env.sh` 只在当前环境检测到 `$CONDA_PREFIX/bin/nvcc` 时临时设置 `CUDA_HOME=$CONDA_PREFIX`，并临时设置 `CC/CXX` 指向环境内的 GCC/G++ 11，不会修改全局 shell 配置。

## 检查环境

```bash
bash scripts/check_env.sh
```

这个脚本会检查 hostname、OS、conda、GPU、driver、`nvidia-smi`、`nvcc`、CUDA 相关环境变量、Nsight 工具，以及 PyTorch/Triton 是否可导入。

## 激活环境

```bash
source scripts/activate_env.sh
```

不要在全局 `.bashrc` 中写死 `CUDA_HOME`。这个 home 目录会在多台服务器之间共享，不同服务器的 driver、CUDA toolkit、`nvcc` 和 Nsight 版本可能不同。

`activate_env.sh` 会根据当前 hostname 设置独立缓存目录：

- `TRITON_CACHE_DIR`
- `TORCH_EXTENSIONS_DIR`
- `CUDA_CACHE_PATH`

脚本优先使用 `/scratch/$USER/cudahannis/$HOSTNAME/...`，如果 `/scratch/$USER` 不存在，则使用 `/tmp/$USER/cudahannis/$HOSTNAME/...`。这样可以避免不同服务器把 Triton/PyTorch 编译缓存混在共享 home 目录里。

## 运行测试

```bash
source scripts/activate_env.sh
python tests/test_triton_add.py
python tests/test_cuda_add.py
python tests/test_activation_24_sparsity.py
python tests/test_nvfp4.py
python tests/test_flash_attention_fp8.py
```

测试会随机生成 CUDA tensor，分别验证 vector add、激活 2:4 稀疏和 NVFP4-style 量化算子。2:4 稀疏和 NVFP4-style 测试会比较 PyTorch reference、Triton kernel 和 CUDA extension 输出。

## 运行激活 2:4 稀疏单脚本

```bash
source scripts/activate_env.sh
python scripts/run_activation_24_sparsity.py
```

算子语义：

- 输入是 CUDA tensor，支持 `float32` 和 `float16`。
- 按最后一维每连续 4 个元素分组。
- 每组保留绝对值最大的 2 个元素，其余置 0。
- 绝对值相等时低索引优先。
- 如果最后一维不是 4 的倍数，内部自动按 0 padding 到 4 的倍数，稀疏化后裁回原 shape。
- 输出是同 shape、同 dtype、同 device 的稠密 tensor。

LLM MLP 中的插入位置示例：

```python
hidden_states = sparsify_before_up_gate(hidden_states, backend="triton")
gate = gate_proj(hidden_states)
up = up_proj(hidden_states)
```

## 运行 NVFP4-style 量化单脚本

```bash
source scripts/activate_env.sh
python scripts/run_nvfp4.py
```

当前服务器是 RTX A6000，compute capability 为 8.6，不是 Blackwell/SM100+，因此这个脚本实现的是可在当前机器跑通的 NVFP4-style 教学路径，不是原生 NVFP4 Tensor Core 路径。

算子语义：

- 输入是 CUDA tensor，支持 `float32` 和 `float16`。
- 按最后一维每连续 16 个元素分组。
- 每组使用一个 `float32` block scale，按 E2M1 FP4 值域 `[0, 0.5, 1, 1.5, 2, 3, 4, 6]` 量化。
- 每两个 FP4 E2M1 code 打包进一个 `uint8`。
- 如果最后一维不是 16 的倍数，内部自动 padding 到 16 的倍数。
- 提供 PyTorch、Triton、CUDA extension 三个 quantize 实现，并用 PyTorch dequantize 做一致性校验。
- 真实 NVIDIA NVFP4 在 Blackwell 上使用 FP8 E4M3 block scale；本项目为了在 A6000 上学习和验证，scale 暂以 `float32` 保存。

## 运行 FP8 causal FlashAttention forward 单脚本

```bash
source scripts/activate_env.sh
python scripts/run_flash_attention_fp8.py
```

当前服务器是 RTX A6000，compute capability 为 8.6，不支持原生 FP8 Tensor Core。本项目的 FP8 FlashAttention 是可在当前机器跑通的教学路径：调用端输入 `float16` Q/K/V，算子内部按每个 `(B, H, S)` row 量化为 `uint8` E4M3 code 和 `float32` scale，再执行 causal attention forward。它用于学习 FP8 存储、scale、causal attention 和 Triton/CUDA extension 组织方式，不代表 Hopper/Blackwell 原生 FP8 FlashAttention 性能。

算子语义：

- 输入 `q/k/v` 是 CUDA contiguous tensor，shape 为 `(B, H, S, D)`，dtype 为 `float16`。
- `D` 支持 `64` 或 `128`。
- 只支持 causal self-attention forward，不支持 backward、dropout、attention bias、varlen、KV cache、GQA/MQA。
- `sm_scale` 默认 `1 / sqrt(D)`。
- 内部使用 per-row E4M3 fake-FP8 量化，最大有限值按 `448.0` 饱和。
- scores 进入 softmax 前按 `float16` 存储语义舍入，输出为 `float16`。
- 提供 PyTorch reference、Triton 和 CUDA extension 三个实现。

## 运行 benchmark

```bash
source scripts/activate_env.sh
python benchmarks/bench_vector_add.py
python benchmarks/bench_flash_attention_fp8.py
```

benchmark 会比较 PyTorch `x + y`、Triton vector add 和 CUDA extension vector add 的平均耗时，并估算读写带宽。这个 benchmark 只用于确认环境和基本性能路径跑通，不追求极限优化。

## Nsight Systems 分析

```bash
source scripts/activate_env.sh
python benchmarks/bench_vector_add.py
nsys profile --force-overwrite=true --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --output=profiles/vector_add_nsys python benchmarks/bench_vector_add.py
```

详细步骤见 [docs/nsight_systems.md](/home/han.li/Code/CUDAHannis/docs/nsight_systems.md)。

## 项目结构

```text
.
├── README.md
├── scripts/
│   ├── check_env.sh
│   ├── activate_env.sh
│   ├── run_activation_24_sparsity.py
│   └── run_nvfp4.py
├── tests/
│   ├── test_triton_add.py
│   ├── test_cuda_add.py
│   ├── test_activation_24_sparsity.py
│   ├── test_nvfp4.py
│   └── test_flash_attention_fp8.py
├── kernels/
│   ├── _cuda_extension.py
│   ├── activation_24/
│   │   ├── activation_24_pytorch.py
│   │   ├── activation_24_triton.py
│   │   ├── activation_24_cuda.py
│   │   ├── activation_24_cuda.cpp
│   │   └── activation_24_cuda_kernel.cu
│   ├── nvfp4/
│   │   ├── nvfp4_pytorch.py
│   │   ├── nvfp4_triton.py
│   │   ├── nvfp4_cuda.py
│   │   ├── nvfp4_cuda.cpp
│   │   └── nvfp4_cuda_kernel.cu
│   ├── flash_attention_fp8/
│   │   ├── flash_attention_fp8_pytorch.py
│   │   ├── flash_attention_fp8_triton.py
│   │   ├── flash_attention_fp8_cuda.py
│   │   ├── flash_attention_fp8_cuda.cpp
│   │   └── flash_attention_fp8_cuda_kernel.cu
│   └── vector_add/
│       ├── vector_add_triton.py
│       ├── vector_add_cuda.py
│       ├── vector_add_cuda.cpp
│       └── vector_add_cuda_kernel.cu
├── docs/
│   └── nsight_systems.md
└── benchmarks/
    └── bench_vector_add.py
```

## 后续学习路线

1. Triton 基础：program id、block、mask、grid、num_warps、autotune。
2. CUDA 基础：thread/block/grid、shared memory、warp、memory coalescing。
3. 性能分析：用 Nsight Systems 看端到端时间线，用 Nsight Compute 看 kernel 指标。
4. 常见算子：reduce、softmax、matmul、layer norm、attention。
5. PyTorch 扩展：`torch.utils.cpp_extension`、自定义 CUDA extension、Triton 和 PyTorch autograd 集成。
