#include <torch/extension.h>

torch::Tensor flash_attention_fp8_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v, double sm_scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &flash_attention_fp8_forward, "FP8 causal FlashAttention forward");
}
