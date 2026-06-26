# CUDAHannis

CUDAHannis 是一个用于学习、实现和验证自定义算子的项目，重点覆盖 PyTorch reference、Triton kernel 与 CUDA extension 的对比实现。仓库中的算子实现集中在 `kernels/` 目录，各算子的基准测试通常放在对应目录的 `benchmark.py` 中。

当前已整理的算子包括：

- vector add：Triton、CUDA extension。
- vector sub：Triton、CUDA extension。
- activation 2:4 sparsity：PyTorch reference、Triton、CUDA extension。
- RMSNorm：PyTorch reference、Triton、CUDA extension。
- sigmoid：PyTorch reference、Triton、CUDA extension。
- sum reduction：PyTorch reference、Triton、CUDA extension。
- NVFP4-style quantization：PyTorch reference、Triton、CUDA extension。
- fake-FP8 causal FlashAttention forward：PyTorch reference、Triton、CUDA extension。

## CUDA Kernel 列表

| CUDA Kernel | Elem DType |
| --- | --- |
| vector_add_f16 | f16 |
| vector_add_f16x2 | f16 |
| vector_add_f16x8 | f16 |
| vector_add_f16x8_pack | f16 |
| vector_sub_f16 | f16 |
| vector_sub_f16x2 | f16 |
| vector_sub_f16x4 | f16 |
| vector_sub_f16x8 | f16 |
| vector_sub_f16x8_pack | f16 |
| activation_24_sparsity_forward | f16 |
| rms_norm_f32 | f32 |
| rms_norm_f32x4 | f32 |
| rms_norm_f16_f16 | f16 |
| rms_norm_f16_f32 | f16 |
| rms_norm_f16x2_f16 | f16 |
| rms_norm_f16x8_f16 | f16 |
| rms_norm_f16x8_f32 | f16 |
| rms_norm_f16x8_pack_f16 | f16 |
| rms_norm_f16x8_pack_f32 | f16 |
| sigmoid_f16 | f16 |
| sigmoid_f16x8 | f16 |
| sigmoid_f16x8_pack | f16 |
| sum_v1 | f32 |
| sum_v2 | f32 |
| sum_v3 | f32 |
| sum_v4 | f32 |

## Benchmark 结果

以下摘要来自：

- `kernels/a01_vector_add/results.txt`
- `kernels/a02_vector_sub/results.txt`
- `kernels/a03_activation_24/results.txt`
- `kernels/a04_rms_norm/results.txt`
- `kernels/a07_sigmoid/results.txt`
- `kernels/a08_sum/results.txt`

这些 benchmark 主要用于观察不同实现路径在固定 shape 下的延迟差异，并快速检查输出是否一致。整体来看，vector add/sub 这类纯 memory-bound elementwise 算子中，Triton、PyTorch 和简单 CUDA extension 的延迟非常接近；pack 或向量化版本在部分 shape 上略有优势。activation 2:4 sparsity 的 CUDA/Triton 实现相比 PyTorch reference 有明显加速，且 `max_abs=0`、`kept=0.5000` 表明结果与 reference 一致。RMSNorm 中，f16 输入使用 f32 累加可以避免大 K 场景下的 f16 overflow，`f16x8_pack` 系列在大 shape 上表现更稳定。sigmoid 的向量化 CUDA 与 Triton 路径和 PyTorch 延迟接近，fp16 输出最大误差约为 `4.883e-04`。sum reduction 中，逐元素全局 `atomicAdd` 的 `sum_v1` 明显较慢，warp/block 规约和 `float4` 读取版本更接近 PyTorch/Triton 的延迟。

### a01_vector_add

完整输出见 `kernels/a01_vector_add/results.txt`。以下为每个 shape 中主要实现的延迟摘要，单位为 ms。

| Shape | f16 | f16x2 | f16x8 | f16x8_pack | triton | torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096x512 | 0.01627588 | 0.01521111 | 0.01564264 | 0.01502919 | 0.01543427 | 0.01548314 |
| 4096x1024 | 0.03746176 | 0.03036928 | 0.03270173 | 0.02946281 | 0.02946305 | 0.02956581 |
| 4096x2048 | 0.05819750 | 0.05735373 | 0.06158519 | 0.05714226 | 0.05706143 | 0.05731678 |
| 4096x4096 | 0.11424136 | 0.11290693 | 0.12581110 | 0.11214662 | 0.11198354 | 0.11335039 |
| 4096x8192 | 0.22237611 | 0.22175407 | 0.22066450 | 0.22321081 | 0.22220755 | 0.22333479 |

### a02_vector_sub

完整输出见 `kernels/a02_vector_sub/results.txt`。以下为每个 shape 中主要实现的延迟摘要，单位为 ms。

| Shape | f16 | f16x2 | f16x8 | f16x8_pack | triton | torch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096x512 | 0.01611805 | 0.01525593 | 0.01661777 | 0.01497674 | 0.01543450 | 0.01548195 |
| 4096x1024 | 0.03015232 | 0.03058648 | 0.03189397 | 0.02946663 | 0.02946019 | 0.02955770 |
| 4096x2048 | 0.05815625 | 0.05807185 | 0.06039834 | 0.05708909 | 0.05705929 | 0.05729651 |
| 4096x4096 | 0.11406183 | 0.11292887 | 0.12922335 | 0.11216903 | 0.11196756 | 0.11339760 |
| 4096x8192 | 0.22217536 | 0.22198129 | 0.25516176 | 0.22249126 | 0.22223568 | 0.22329855 |

### a03_activation_24

完整输出见 `kernels/a03_activation_24/results.txt`。以下为每个 shape 中主要实现的延迟摘要，单位为 ms。

| Shape | torch | cuda | triton_v1 | triton_v2 | triton_v6 | max_abs | kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096x512 | 1.91493500 | 0.01153777 | 0.01500234 | 0.01517086 | 0.01292666 | 0.000e+00 | 0.5000 |
| 4096x1024 | 3.82785224 | 0.02040074 | 0.02081631 | 0.02145310 | 0.02049320 | 0.000e+00 | 0.5000 |
| 4096x2048 | 7.20833890 | 0.03877959 | 0.03947358 | 0.04009204 | 0.03890813 | 0.000e+00 | 0.5000 |
| 4096x4096 | 14.27223850 | 0.07522552 | 0.07755497 | 0.07753914 | 0.07536758 | 0.000e+00 | 0.5000 |
| 4096x8192 | 28.79997090 | 0.14845902 | 0.15557472 | 0.15598029 | 0.14875578 | 0.000e+00 | 0.5000 |

### a04_rms_norm

完整输出见 `kernels/a04_rms_norm/results.txt`。以下为每个 shape 中主要实现的延迟摘要，单位为 ms。

| Shape | f32 | f32x4 | f32_torch |
| --- | ---: | ---: | ---: |
| 4096x512 | 0.02202487 | 0.02050757 | 0.07267761 |
| 4096x1024 | 0.05628395 | 0.03893900 | 0.14056492 |
| 4096x2048 | - | 0.07592463 | 0.27227163 |
| 4096x4096 | - | 0.15078878 | 0.53322887 |

| Shape | f16x8f16 | f16x8f32 | f16x8packf16 | f16x8packf32 | f16_torch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096x512 | 0.00928044 | 0.00923109 | 0.00930381 | 0.00923514 | 0.03878641 |
| 4096x1024 | 0.02037311 | 0.02022886 | 0.02046728 | 0.02042818 | 0.07606101 |
| 4096x2048 | 0.03868556 | 0.03872442 | 0.03900266 | 0.03897095 | 0.14600539 |
| 4096x4096 | 0.07539630 | 0.07543254 | 0.07596612 | 0.07607031 | 0.28232479 |
| 4096x8192 | 0.17227817 | 0.16541743 | 0.15084887 | 0.15088725 | 0.53825569 |
| 8192x8192 | 0.34065270 | 0.33396292 | 0.29863858 | 0.29865313 | 1.06043911 |

### a07_sigmoid

完整输出见 `kernels/a07_sigmoid/results.txt`。以下为每个 shape 中主要实现的延迟摘要，单位为 ms。

| Shape | torch | f16 | f16x8 | f16x8_pack | triton | max_abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096x512 | 0.01129363 | 0.01366665 | 0.00916062 | 0.00938406 | 0.01291644 | 4.883e-04 |
| 4096x1024 | 0.02099489 | 0.03614181 | 0.02052204 | 0.02063214 | 0.02057587 | 4.883e-04 |
| 4096x2048 | 0.04013416 | 0.04661263 | 0.03901708 | 0.03920933 | 0.03916700 | 4.883e-04 |
| 4096x4096 | 0.07720426 | 0.09106254 | 0.07571516 | 0.07627705 | 0.07604074 | 4.883e-04 |
| 4096x8192 | 0.15094839 | 0.17631662 | 0.14998780 | 0.15056026 | 0.15007334 | 4.883e-04 |

### a08_sum

完整输出见 `kernels/a08_sum/results.txt`。`sum_v1` 使用每元素全局 `atomicAdd`，benchmark 中单独使用 `iters=10`；其他实现使用 `iters=100`。

| Shape | torch | v1 | v2 | v3 | v4 | triton |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096x512 | 0.01530097 | 3.47130250 | 0.02888352 | 0.01746334 | 0.01279195 | 0.05288825 |
| 4096x1024 | 0.02400474 | 6.93783720 | 0.05348507 | 0.03107473 | 0.02154411 | 0.05086244 |
| 4096x2048 | 0.04142580 | 13.51017010 | 0.09761085 | 0.05497749 | 0.03884294 | 0.05209496 |
| 4096x4096 | 0.07613867 | 26.22011730 | 0.19120885 | 0.10618507 | 0.07381731 | 0.07664883 |
| 4096x8192 | 0.16181348 | 52.43447490 | 0.37838473 | 0.20855838 | 0.14364662 | 0.14697498 |
