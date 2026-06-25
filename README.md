# CUDAHannis

CUDAHannis 是一个用于学习和验证 PyTorch、Triton 与 CUDA extension 自定义算子的项目。仓库将算子实现集中在 `kernels/`，将正确性、性能和真实模型集成评测集中在 `eval/`。

当前可调用的算子包括：

- vector add：Triton、CUDA extension。
- activation 2:4 sparsity：PyTorch reference、Triton、CUDA extension。
- NVFP4-style quantization：PyTorch reference、Triton、CUDA extension。
- fake-FP8 causal FlashAttention forward：PyTorch reference、Triton、CUDA extension。

`kernels/rms_norm/rms_norm_f16_f32.cu` 和 `kernels/block_all_reduce/block_all_reduce.cn` 是保留的独立实验源码，目前没有 Python loader 或标准化 eval。其中 RMSNorm 使用标量 scale，不能直接替换 Qwen3 的逐通道 RMSNorm weight。仓库当前没有自定义 SiLU/Swish 或 NMS 实现。

## 环境

项目当前验证环境：

- 服务器：`shiva`
- conda 环境：`vllm-cu129`
- PyTorch：`2.11.0+cu130`
- Triton：`3.6.0`
- CUDA extension toolkit：CUDA 13（`nvidia-cuda-nvcc 13.2.78`，pip 安装于 `site-packages/nvidia/cu13`）
- CUDA extension host compiler：系统 GCC/G++ 13.3.0

检查并激活环境：

```bash
bash scripts/check_env.sh
source scripts/activate_env.sh
```

`activate_env.sh` 会把 `CUDA_HOME` 指向 pip 安装的 nvcc（`site-packages/nvidia/cu13`）、使用系统 GCC/G++ 作为 host compiler，并按 hostname 隔离以下编译缓存：

- `TRITON_CACHE_DIR`
- `TORCH_EXTENSIONS_DIR`
- `CUDA_CACHE_PATH`

不要在共享 home 的全局 shell 配置中写死 `CUDA_HOME`。

所有 Hugging Face 操作固定使用：

```bash
export HF_HOME=/mnt/workspace/users/han.li/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
```

Qwen3 评测入口也会在导入 Hugging Face 相关库前设置这些路径。

## 项目结构

```text
.
├── kernels/
│   ├── _cuda_extension.py
│   ├── activation_24/
│   ├── block_all_reduce/
│   ├── flash_attention_fp8/
│   ├── nvfp4/
│   ├── rms_norm/
│   └── vector_add/
├── eval/
│   ├── common.py
│   ├── correctness/
│   ├── benchmarks/
│   └── qwen3_8b/
│       ├── remote_code/
│       ├── evaluate_activation_24.py
│       ├── run_activation_24_comparison.sh
│       └── test_remote_code.py
├── scripts/
│   ├── activate_env.sh
│   └── check_env.sh
├── docs/
│   └── nsight_systems.md
├── AGENTS.md
└── README.md
```

每个可调用算子目录包含 Python public API、PyTorch reference（适用时）、Triton 实现和 CUDA/C++ extension 源码。测试、benchmark 或模型集成逻辑不得放进 `kernels/`。

## 编译和加载 CUDA extension

CUDA extension 使用 `torch.utils.cpp_extension.load` 按需 JIT 编译，不需要单独执行 setup/build 命令。首次调用 CUDA backend 时会从对应算子目录读取 `.cpp`/`.cu` 文件，并将产物写入 `TORCH_EXTENSIONS_DIR`。

```bash
source scripts/activate_env.sh
python - <<'PY'
import torch
from kernels.vector_add import cuda_add

x = torch.randn(1024, device="cuda")
y = torch.randn_like(x)
print(cuda_add(x, y))
PY
```

Triton backend 在首次调用时自动 JIT 编译并使用 `TRITON_CACHE_DIR`。

## 单算子正确性测试

运行全部单算子测试：

```bash
source scripts/activate_env.sh
pytest -q eval/correctness
```

运行指定算子：

```bash
pytest -q eval/correctness/test_activation_24.py
pytest -q eval/correctness/test_flash_attention_fp8.py
pytest -q eval/correctness/test_nvfp4.py
pytest -q eval/correctness/test_vector_add.py
```

测试使用参数化 dtype、shape 和 batch size，并覆盖 padding、非对齐维度、dispatch、边界值和非法输入。PyTorch 实现作为 reference；精确算子检查完全一致，FP8 attention 使用明确的数值容差。

## 单算子 benchmark

每个 benchmark 支持重复使用 `--shape` 和 `--dtype`，并支持 `--warmup`、`--repeat`：

```bash
source scripts/activate_env.sh
python -m eval.benchmarks.benchmark_vector_add
python -m eval.benchmarks.benchmark_activation_24
python -m eval.benchmarks.benchmark_nvfp4
python -m eval.benchmarks.benchmark_flash_attention_fp8
```

自定义短测试示例：

```bash
python -m eval.benchmarks.benchmark_activation_24 \
  --shape 1x16x4096 \
  --dtype float16 \
  --warmup 2 \
  --repeat 5
```

输出包含 backend、shape、dtype、平均延迟、相对 PyTorch speedup、最大/平均绝对误差。vector add 额外输出带宽；NVFP4 额外输出 packed mismatch、scale 误差和重建 MSE。

## Qwen3-8B activation 2:4 替换评测

Qwen/Qwen3-8B 的 Hugging Face snapshot 不包含原生 remote-code Python 文件，它使用 Transformers 内置的 Qwen3 实现。此项目不会修改模型 snapshot，而是创建一个位于 `$HF_HOME/overlays` 的本地 overlay：

1. 模型权重和 tokenizer 文件通过 symlink 复用原 snapshot。
2. overlay 的 `config.json` 将 `AutoModelForCausalLM` 指向项目提供的 remote code。
3. 自定义 `Qwen3Activation24MLP` 只在 `mlp_input` 和 `down_input` 插入 activation 2:4 sparsity。
4. 模型原始 SiLU、RMSNorm、attention 和权重加载逻辑保持不变。

先验证 overlay 和 tiny-model 集成：

```bash
source scripts/activate_env.sh
pytest -q eval/qwen3_8b/test_remote_code.py
python -m eval.qwen3_8b.evaluate_activation_24 --prepare-only
```

单个变体快速测试：

```bash
python -m eval.qwen3_8b.evaluate_activation_24 \
  --variant triton \
  --tasks boolq \
  --limit 10 \
  --output-json results/qwen3_8b/triton_smoke.json
```

在一张 GPU 上串行对比 baseline、PyTorch reference 和 Triton：

```bash
GPU=0 \
EVAL_TASKS=boolq \
EVAL_LIMIT=10 \
bash eval/qwen3_8b/run_activation_24_comparison.sh
```

完整默认任务集合运行：

```bash
GPU=0 bash eval/qwen3_8b/run_activation_24_comparison.sh
```

默认任务为 `hellaswag,piqa,winogrande,arc_challenge,arc_easy,boolq`。结果和日志写入 `results/qwen3_8b/`，该目录不纳入版本控制。

## eval 文件用途

| 文件 | 用途 |
| --- | --- |
| `eval/common.py` | 统一 shape/dtype 参数解析、CUDA Event 计时和误差输出。 |
| `eval/correctness/test_vector_add.py` | vector add 多 dtype/shape、CUDA/Triton correctness。 |
| `eval/correctness/test_activation_24.py` | 2:4 稀疏 correctness、tie-break、padding 和 backend dispatch。 |
| `eval/correctness/test_nvfp4.py` | packed code、scale、dequant 和 padding correctness。 |
| `eval/correctness/test_flash_attention_fp8.py` | 多 batch/head/sequence/head-dim correctness 和输入校验。 |
| `eval/benchmarks/benchmark_vector_add.py` | vector add 延迟、误差和有效带宽。 |
| `eval/benchmarks/benchmark_activation_24.py` | 2:4 稀疏 PyTorch/Triton/CUDA 对比。 |
| `eval/benchmarks/benchmark_nvfp4.py` | NVFP4-style quantize 延迟和量化误差。 |
| `eval/benchmarks/benchmark_flash_attention_fp8.py` | fake-FP8 attention 延迟和误差。 |
| `eval/qwen3_8b/evaluate_activation_24.py` | Qwen3-8B 单变体 lm-eval 与 shape 统计。 |
| `eval/qwen3_8b/run_activation_24_comparison.sh` | 单 GPU 串行运行并合并三个变体。 |
| `eval/qwen3_8b/test_remote_code.py` | tiny Qwen3 模型替换公式、统计和算子 benchmark 测试。 |

新增算子时，先创建 `kernels/<operator>/` 并通过其 `__init__.py` 暴露稳定接口，再分别复制一个 correctness 和 benchmark 文件，使用 `eval.common` 的统一参数、计时和误差工具。

## Profiling

Nsight Systems 的采集和分析流程见 [docs/nsight_systems.md](docs/nsight_systems.md)。
