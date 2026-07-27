## 可能的运行环境

目前常用的运行环境包括：

Blackwell
- 服务器 `brahma`：conda 环境 `vllm-cu129`
- 服务器 `parvati`：conda 环境 `vllm-cu129`
Ampere
- 服务器 `ISLAB`（192.168.3.191）：conda 环境 `vllm192`

```bash
export HF_HOME=/mnt/workspace/users/han.li/hf_home
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_ASSETS_CACHE=$HF_HOME/assets
```
