version1

Treat each row's last dimension as groups of 4 values and launch Triton programs over rows and group blocks.
For each 4-value group, compare absolute values with deterministic lower-index tie breaking to keep rank < 2.
Store kept values unchanged and write zero for the other lanes, masking padded tail elements.

Triton compare_single, shape (1, 1024, 12288):
float32: max error 0.00000000e+00, 0.1660 ms.
float16: max error 0.00000000e+00, 0.0823 ms.

version2

Add a full-block fast path for Qwen3.5-9B's divisible intermediate size, removing tail masks and using pairwise top-2 selection.
Use 64 groups per program with 2 warps to reduce launch granularity and improve fp16 occupancy.

Triton compare_single, shape (1, 1024, 12288):
float32: max error 0.00000000e+00, 0.1624 ms.
float16: max error 0.00000000e+00, 0.0776 ms.

version3

Add a contiguous-element fast path for fully aligned tensors, trading redundant neighbor loads for coalesced element-wise reads/stores.
Use block size 128 for float32 and 256 for float16; keep the older masked/group kernels for irregular shapes.

Triton compare_single, shape (1, 1024, 12288):
float32: max error 0.00000000e+00, 0.1465 ms.
float16: max error 0.00000000e+00, 0.0756 ms.

version4

Keep the contiguous fast path but remove the redundant self-value load; select the lane value from the already loaded group values.
Compute rank only for the current lane using lane-aware tie breaking, while preserving the version3 block sizes.

Triton compare_single, shape (1, 1024, 12288):
float32: max error 0.00000000e+00, 0.1463 ms.
float16: max error 0.00000000e+00, 0.0756 ms.

version5

Compute the contiguous fast-path lane from local offsets to simplify address arithmetic.
Retune the contiguous launch shape: float32 uses block size 512 with 2 warps, float16 keeps block size 256 with 4 warps.

Triton compare_single, shape (1, 1024, 12288):
float32: max error 0.00000000e+00, 0.1459 ms.
float16: max error 0.00000000e+00, 0.0748 ms.

version6

Keep the version5 contiguous kernel parameters, but skip computing fallback row/group/grid metadata on the aligned fast path.
2D grouped-load and cache-modifier experiments were slower or unstable, so they were not kept.

Triton compare_single, shape (1, 1024, 12288):
float32: max error 0.00000000e+00, 0.1468 ms.
float16: max error 0.00000000e+00, 0.0748 ms.
