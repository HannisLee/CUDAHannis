## 可能的运行环境

目前常用的运行环境包括：

- 服务器 `shiva`：conda 环境 `vllm-cu129`
- 服务器 `ISLAB`（192.168.3.191）：conda 环境 `vllm192`

## Hugging Face 缓存路径

如果在服务器 `shiva` 上运行，所有涉及 `HF_HOME` 的操作都需要显式指定到 `/mnt/workspace/users/han.li/hf_home`：

```bash
export HF_HOME=/mnt/workspace/users/han.li/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
```
