## 当前环境

- 服务器：`shiva`
- conda 环境：`triton-cu118`
- PyTorch CUDA runtime：`cu118`
- Triton 版本：由 PyTorch 依赖固定，目前为 `3.3.1`
- CUDA extension 编译依赖：`cuda-nvcc/cuda-cudart/cuda-cudart-dev/cuda-cccl 11.8.89`，安装在 `triton-cu118` 环境内
- CUDA extension host compiler：`gcc_linux-64/gxx_linux-64 11.4.0`

## Hugging Face 缓存路径

所有涉及到 `HF_HOME` 的操作，都要指定到 `/mnt/workspace/users/han.li/hf_home`：

```bash
export HF_HOME=/mnt/workspace/users/han.li/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
```
