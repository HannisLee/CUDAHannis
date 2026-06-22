#include <torch/extension.h>

std::vector<torch::Tensor> nvfp4_quantize_forward(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &nvfp4_quantize_forward, "NVFP4-style quantize forward");
}
