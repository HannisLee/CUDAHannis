#include <torch/extension.h>

torch::Tensor activation_24_sparsity_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &activation_24_sparsity_forward, "Activation 2:4 sparsity forward");
}
