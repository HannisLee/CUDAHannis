// Bindings for the activation 2:4 sparsity CUDA kernels, versions 1-4.
// All four forward functions are linked from the per-version .cu translation
// units and exposed as one extension module.

#include <torch/extension.h>

torch::Tensor activation_24_sparsity_forward_v1(torch::Tensor x);
torch::Tensor activation_24_sparsity_forward_v2(torch::Tensor x);
torch::Tensor activation_24_sparsity_forward_v3(torch::Tensor x);
torch::Tensor activation_24_sparsity_forward_v4(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_v1", &activation_24_sparsity_forward_v1,
        "Activation 2:4 sparsity forward (version1: masked group kernel)");
  m.def("forward_v2", &activation_24_sparsity_forward_v2,
        "Activation 2:4 sparsity forward (version2: aligned mask-free fast path)");
  m.def("forward_v3", &activation_24_sparsity_forward_v3,
        "Activation 2:4 sparsity forward (version3: vectorized contiguous fast path)");
  m.def("forward_v4", &activation_24_sparsity_forward_v4,
        "Activation 2:4 sparsity forward (version4: vectorized + branchless selection)");
}
