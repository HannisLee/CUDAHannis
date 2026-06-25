CUDA_VISIBLE_DEVICES=0 \
vllm serve /mnt/workspace/users/han.li/models/Qwen--Qwen3.6-35B-A3B \
  --served-model-name qwen36-35b-a3b \
  --host 0.0.0.0 \
  --port 8081 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --language-model-only \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 32  \
  > /home/han.li/Code/CUDAHannis/Qwen3.6/Qwen3.6.log 2>&1 &