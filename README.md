# CUDAHannis

CUDAHannis 是一个用于学习、实现和验证自定义算子的项目，重点覆盖 PyTorch reference、Triton kernel 与 CUDA extension 的对比实现。仓库中的算子实现集中在 `kernels/` 目录，各算子的基准测试通常放在对应目录的 `benchmark.py` 中。

当前已整理的算子包括：

- vector add：Triton、CUDA extension。
- vector sub：Triton、CUDA extension。
- activation 2:4 sparsity：PyTorch reference、Triton、CUDA extension。
- NVFP4-style quantization：PyTorch reference、Triton、CUDA extension。
- fake-FP8 causal FlashAttention forward：PyTorch reference、Triton、CUDA extension。
