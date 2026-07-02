# ENVIRONMENT.md — Agent 环境复现指令

> **本文件是给 AI agent(Claude Code 等)的提示词**,用于在任意一台机器上把
> CUDAHannis 的运行环境准备到位。如果你是 agent,请**从头到尾按顺序执行**,
> 不要跳过检测步骤、不要无脑重装。

---

## 0. 执行原则(必须遵守)

1. **检测优先(Detect-first)**:动手安装前,必须先探明本机已有哪些资源(GPU、驱动、CUDA toolkit、conda 环境、torch/triton、g++、ninja、cmake)。
2. **复用优先(Reuse-first)**:本机若已有满足要求的环境,**直接复用**,不要新建、不要重装。
3. **最小改动**:只补装缺失的组件,不改动用户已有的可用环境。
4. **全程汇报**:把每一步检测到的版本、最终选择复用/新建哪个环境、为什么,清楚地告诉用户。

**Canonical 环境名是 `cudahannis`**。新建环境时一律用这个名字。但若本机已有满足要求的环境(无论叫什么名字),优先复用那个(见 §3 决策树)。

---

## 1. 依赖与版本约束

| 组件 | 作用 | 一般要求 | 备注 |
| --- | --- | --- | --- |
| NVIDIA GPU + 驱动 | 跑 kernel | 任意 CUDA-capable GPU | fp8/fp4 算子需 Ada(sm89+)/Hopper(sm90) |
| **CUDA Toolkit (nvcc)** | 编译 `.cu` | ≥ 11.8(用到 `cuda_fp8.h` 时) | 由 `CUDA_HOME` 指向,通常 `/usr/local/cuda` |
| **PyTorch(CUDA 版)** | 提供 `torch/extension.h` + `cpp_extension.load()` JIT | 必须 **CUDA build**(`+cuXXX`),2.x | **CPU-only 版不能用** |
| **triton** | 跑 `*_triton.py` | 3.x | 纯 CUDA 算子不强依赖 |
| g++ / gcc | nvcc 的 host 编译器 | CUDA toolkit 支持的版本(gcc 11/12 安全) | nvcc 启动会校验 gcc 版本 |
| Python | 跑 benchmark/封装 | 3.9 ~ 3.12 | |
| ninja | `load()` 并行 JIT 编译 | 任意近期版 | 没有会退回 setuptools,变慢 |
| cmake | cpp_extension 部分路径使用 | 任意近期版 | |

### 三条硬约束(踩坑来源)

1. **torch 必须是 CUDA 版**。`.cu` 里 `#include <torch/extension.h>` 的头随 pip 装的 torch 提供;编出来的 `.so` 要链接 `libtorch_cuda.so`,CPU 版 torch 没有。
2. **nvcc 大版本尽量与 `torch.version.cuda` 对齐**。例:torch `cu126`(CUDA 12.6)配 CUDA 12.x 的 nvcc 最稳。跨大版本(如 nvcc 13 + torch cu126)靠向后兼容,能跑但 PyTorch 会 warning,**非必要不混用**。
3. **gcc 版本不能超过 CUDA toolkit 的上限**。装错 nvcc 会直接报错。

### 头文件来源(排查编译错误时用)

- `cuda_fp16.h` / `cuda_bf16.h` / `cuda_fp8.h` / `cuda_runtime.h` → 来自 **CUDA Toolkit**(`$CUDA_HOME/include`)
- `torch/extension.h` / `torch/types.h` → 来自 **PyTorch** 安装目录

---

## 2. Step 1 — 检测硬件与工具链

先把下面整段跑掉,把输出记下来汇报给用户:

```bash
echo "== GPU/驱动 =="; nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
echo "== nvcc / CUDA =="; nvcc --version | tail -2; echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "== host 编译器 =="; gcc --version | head -1; g++ --version | head -1
echo "== python =="; python --version
echo "== 构建工具 =="; ninja --version 2>/dev/null || echo "ninja: 无"; cmake --version 2>/dev/null | head -1 || echo "cmake: 无"
echo "== conda 环境列表 =="; conda env list
```

判定:
- 没有 `nvidia-smi` / 没有 GPU → 停,告诉用户这台机器跑不了 GPU kernel。
- 没有 `nvcc` 且 `CUDA_HOME` 为空 → 需要装 CUDA Toolkit(见 §4 末)。
- 已有 `nvcc` → 记下其**大版本**(如 release 13.2 → 大版本 13),后面选 torch wheel 要用到。

---

## 3. Step 2 — 检测已有 conda 环境并逐个探测

**这是"优先检测本机已有环境"的核心步骤。** 逐个环境探测 torch/triton:

```bash
for e in $(conda env list | grep -vE '^#|^\s*$' | awk '{print $1}'); do
  echo "== $e =="
  conda run -n "$e" python -c \
    "import torch,triton;print('torch',torch.__version__,'| cuda',torch.version.cuda,'| gpu_ok',torch.cuda.is_available(),'| triton',triton.__version__)" \
    2>/dev/null || echo "(无 torch/triton)"
done
```

**"满足要求"的判定标准**(一个环境满足下列全部即可复用):

1. `import torch, triton` 都成功;
2. `torch.cuda.is_available()` 为 `True`;
3. `torch.version.cuda` 与系统 nvcc **大版本一致**(一致最理想;不一致但能用也算可接受,需提醒用户)。

---

## 4. Step 3 — 复用 / 创建 决策树

按顺序判断,**命中即停**:

```
1) 任意一个已有环境(含当前激活的)满足 §3 标准?
   → 是:直接复用它。告诉用户"复用现有环境 X",后续命令都用 `conda run -n X`
        或激活 X 后执行。不要新建 cudahannis。结束。

2) 名为 cudahannis 的环境已存在但不完整(缺 torch/triton 或 cuda 不可用)?
   → 是:进入 cudahannis,只补装缺失的包(见 Step 4)。结束。

3) 本机没有任何满足要求的环境?
   → 新建 cudahannis(见 Step 4)。
```

> 说明:原则上是"能复用就不新建"。若用户明确要求所有机器都统一用 `cudahannis`
> 这个名字,可对已满足要求的旧环境做克隆:
> `conda create --name cudahannis --clone <旧环境>`(硬链接,不占额外空间)。
> 否则直接复用旧环境即可。

---

## 5. Step 4 — 安装(仅在需要时)

### 4.1 新建 `cudahannis`

```bash
conda create -n cudahannis python=3.12 -y
conda activate cudahannis          # 或后续用 conda run -n cudahannis
```

### 4.2 装 torch(CUDA 版)

**先按 §2 的 nvcc 大版本选 cuXXX wheel**,对齐优先:

| 系统 nvcc 大版本 | 推荐 torch wheel index-url |
| --- | --- |
| CUDA 12.x | `https://download.pytorch.org/whl/cu126`(或 cu128) |
| CUDA 13.x | 暂用最新 cu12x wheel(靠运行时兼容,本机即此情况) |
| CUDA 11.x | `https://download.pytorch.org/whl/cu118` |

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

### 4.3 装 triton 与构建工具

```bash
pip install triton ninja cmake
```

### 4.4 没有 nvcc / CUDA Toolkit 时

CUDA Toolkit 需从 NVIDIA 官网装(不在 conda/pip 范围),装好后确保:

```bash
export CUDA_HOME=/usr/local/cuda          # 指向 toolkit 根
export PATH=$CUDA_HOME/bin:$PATH
```

`nvcc --version` 能出版本才算成功。

---

## 6. Step 5 — 验证

```bash
# 1) 基础导入与 CUDA 可用性
python -c "import torch,triton; assert torch.cuda.is_available(); \
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| triton', triton.__version__, \
'| gpu', torch.cuda.get_device_name(0))"
```

```bash
# 2) JIT 编译冒烟测试:随便挑一个算子跑 benchmark(首次会现场编译 .cu)
cd kernels/a06_gelu && python benchmark.py
```

两条都通过(冒烟测试打出 `time:` 行、`max_abs` 很小)即环境就绪。

---

## 7. Step 6 — 跑某个算子

```bash
cd kernels/<算子目录>      # 例:a01_vector_add / a06_gelu / a08_sum ...
python benchmark.py        # 首次慢(JIT 编译),之后命中 ~/.cache/torch_extensions 缓存秒开
```

每个算子目录通常含:`*.cu`(CUDA extension)、`*_triton.py`、`*_pytorch.py`(reference)、`benchmark.py`、`results.txt`。

---

## 8. 注意事项 / 已知坑

- **JIT 缓存**:`torch.utils.cpp_extension.load()` 把编出的 `.so` 缓存在 `~/.cache/torch_extensions/`。改了 `.cu` 或编译参数会自动重编;若遇到诡异的旧代码残留,删掉该缓存目录再跑。
- **`--use_fast_math`**:`benchmark.py` 里开了快速数学,`tanhf` 等与 PyTorch 参考会有 ~5e-6(fp32)/ ~1e-3(fp16) 的数值差,属正常。
- **跨大版本 CUDA**:nvcc 与 `torch.version.cuda` 大版本不一致时 PyTorch 会 warning,尽量对齐。
- **fp8 / fp4 算子**(`kernels/flash_attention_fp8`、`kernels/nvfp4`):需要 Ada(sm89)/ Hopper(sm90) 及以上 GPU 才能真正运行,其它机器只能编译不能跑。
- **服务器 `shiva` 的 HF_HOME**(仅当用到 HuggingFace 模型/数据时,纯算子 benchmark 不需要):

  ```bash
  export HF_HOME=/mnt/workspace/users/han.li/hf_home
  export HF_HUB_CACHE=$HF_HOME/hub
  export HF_DATASETS_CACHE=$HF_HOME/datasets
  export HF_ASSETS_CACHE=$HF_HOME/assets
  ```

- **历史环境名**:服务器上可能存在 `vllm192`(ISLAB)、`vllm-cu129`(shiva)等旧环境。若它们满足 §3 标准,按"复用优先"直接用;新建则统一叫 `cudahannis`。

---

## 附录:本机(开发机 lihan)实测基线

作为"满足要求的环境"样例,供对照:

| 组件 | 实测版本 |
| --- | --- |
| GPU / 驱动 / sm | RTX 3090 Ti / 595.58.03 / 8.6 |
| CUDA Toolkit (nvcc) | 13.2(`CUDA_HOME=/usr/local/cuda`) |
| PyTorch | 2.10.0+cu126(`torch.version.cuda=12.6`,跨大版本但可用) |
| triton | 3.6.0 |
| gcc/g++ | 11.4.0 |
| Python | 3.12.13 |
| ninja / cmake | 1.13 / 3.22.1 |
| 现有可用 conda 环境 | `vllm192`(已含 torch+cuda+triton,本机可直接复用) |
