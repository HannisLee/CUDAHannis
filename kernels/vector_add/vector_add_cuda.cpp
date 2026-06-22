#include <torch/extension.h>

torch::Tensor cuda_add_forward(torch::Tensor x, torch::Tensor y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cuda_add_forward, "CUDA vector add forward");
}
