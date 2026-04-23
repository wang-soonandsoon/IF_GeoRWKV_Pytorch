#include <torch/extension.h>

#include <vector>

// CUDA forward declaration
at::Tensor wkv_dynamic_cuda_forward(
    const at::Tensor& r,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& time_decay,
    const at::Tensor& time_first,
    const at::Tensor& log_gate);

at::Tensor wkv_dynamic_forward(
    const at::Tensor& r,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& time_decay,
    const at::Tensor& time_first,
    const at::Tensor& log_gate) {
  TORCH_CHECK(r.is_cuda(), "r must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
  TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");
  TORCH_CHECK(time_decay.is_cuda(), "time_decay must be a CUDA tensor");
  TORCH_CHECK(time_first.is_cuda(), "time_first must be a CUDA tensor");
  TORCH_CHECK(log_gate.is_cuda(), "log_gate must be a CUDA tensor");

  TORCH_CHECK(r.dim() == 3, "r must be [B, L, D]");
  TORCH_CHECK(k.dim() == 3, "k must be [B, L, D]");
  TORCH_CHECK(v.dim() == 3, "v must be [B, L, D]");
  TORCH_CHECK(log_gate.dim() == 3 && log_gate.size(2) == 1, "log_gate must be [B, L, 1]");
  TORCH_CHECK(r.sizes() == k.sizes() && k.sizes() == v.sizes(), "r/k/v shapes must match");
  TORCH_CHECK(r.size(0) == log_gate.size(0) && r.size(1) == log_gate.size(1), "sequence dims must match log_gate");

  TORCH_CHECK(r.scalar_type() == k.scalar_type() && k.scalar_type() == v.scalar_type(), "r/k/v dtypes must match");
  TORCH_CHECK(
      r.scalar_type() == at::kHalf || r.scalar_type() == at::kBFloat16 || r.scalar_type() == at::kFloat,
      "r/k/v must be float16, bfloat16, or float32");

  TORCH_CHECK(time_decay.dim() == 1, "time_decay must be [D]");
  TORCH_CHECK(time_first.dim() == 1, "time_first must be [D]");
  TORCH_CHECK(time_decay.size(0) == r.size(2), "time_decay size must equal D");
  TORCH_CHECK(time_first.size(0) == r.size(2), "time_first size must equal D");
  TORCH_CHECK(time_decay.scalar_type() == at::kFloat, "time_decay must be float32");
  TORCH_CHECK(time_first.scalar_type() == at::kFloat, "time_first must be float32");
  TORCH_CHECK(log_gate.scalar_type() == at::kFloat, "log_gate must be float32");

  TORCH_CHECK(r.is_contiguous(), "r must be contiguous");
  TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
  TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
  TORCH_CHECK(time_decay.is_contiguous(), "time_decay must be contiguous");
  TORCH_CHECK(time_first.is_contiguous(), "time_first must be contiguous");
  TORCH_CHECK(log_gate.is_contiguous(), "log_gate must be contiguous");

  return wkv_dynamic_cuda_forward(r, k, v, time_decay, time_first, log_gate);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &wkv_dynamic_forward, "Dynamic WKV forward (CUDA)");
}
