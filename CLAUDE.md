## 当前环境

- 服务器：`shiva`
- conda 环境：`vllm-cu129`
- PyTorch CUDA runtime：`cu130`
- Triton 版本：`3.6.0`
- CUDA extension 编译依赖：`nvidia-cuda-nvcc 13.2.78 / nvidia-cuda-cccl 13.3.3.3.1 / nvidia-cuda-runtime 13.0.96`，通过 pip 安装在 `vllm-cu129` 环境内（`site-packages/nvidia/cu13`）
- CUDA extension host compiler：系统 `gcc/g++ 13.3.0`（环境内无 conda gcc）

## Hugging Face 缓存路径

所有涉及到 `HF_HOME` 的操作，都要指定到 `/mnt/workspace/users/han.li/hf_home`：

```bash
export HF_HOME=/mnt/workspace/users/han.li/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
```
