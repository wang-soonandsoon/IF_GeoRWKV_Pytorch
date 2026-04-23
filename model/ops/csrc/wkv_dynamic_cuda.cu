#include <torch/extension.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t x) {
  return static_cast<float>(x);
}

__device__ __forceinline__ float neg_softplus(float x) {
  float m = fmaxf(x, 0.0f);
  return -(m + log1pf(expf(-fabsf(x))));
}

template <typename scalar_t>
__global__ void wkv_dynamic_forward_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const float* __restrict__ time_decay,
    const float* __restrict__ time_first,
    const float* __restrict__ log_gate,
    scalar_t* __restrict__ out,
    int B,
    int L,
    int D) {
  const int bd = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = B * D;
  if (bd >= total) {
    return;
  }

  const int b = bd / D;
  const int d = bd - b * D;

  float aa = 0.0f;
  float bb = 0.0f;
  float pp = -1.0e30f;

  const float base_log_decay = neg_softplus(time_decay[d]);
  const float tf = time_first[d];

  for (int t = 0; t < L; ++t) {
    const int idx = (b * L + t) * D + d;
    const int lg_idx = (b * L + t);

    const float kt = to_float(k[idx]);
    const float vt = to_float(v[idx]);
    const float rt = to_float(r[idx]);
    const float gt = log_gate[lg_idx];

    const float ww = tf + kt;
    const float p = fmaxf(pp, ww);
    const float e1 = expf(pp - p);
    const float e2 = expf(ww - p);
    const float denom = e1 * bb + e2 + 1.0e-6f;
    const float wkv = (e1 * aa + e2 * vt) / denom;
    out[idx] = static_cast<scalar_t>(wkv * rt);

    const float ww2 = (base_log_decay + gt) + pp;
    const float p2 = fmaxf(ww2, kt);
    const float f1 = expf(ww2 - p2);
    const float f2 = expf(kt - p2);
    aa = f1 * aa + f2 * vt;
    bb = f1 * bb + f2;
    pp = p2;
  }
}

}  // namespace

at::Tensor wkv_dynamic_cuda_forward(
    const at::Tensor& r,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& time_decay,
    const at::Tensor& time_first,
    const at::Tensor& log_gate) {
  const auto B = static_cast<int>(r.size(0));
  const auto L = static_cast<int>(r.size(1));
  const auto D = static_cast<int>(r.size(2));

  at::Tensor out = at::zeros_like(r);

  const int threads = 256;
  const int total = B * D;
  const int blocks = (total + threads - 1) / threads;

  const c10::cuda::CUDAGuard device_guard(r.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(r.get_device()).stream();

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, r.scalar_type(), "wkv_dynamic_cuda_forward", [&] {
    wkv_dynamic_forward_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        r.data_ptr<scalar_t>(),
        k.data_ptr<scalar_t>(),
        v.data_ptr<scalar_t>(),
        time_decay.data_ptr<float>(),
        time_first.data_ptr<float>(),
        log_gate.data_ptr<float>(),
        out.data_ptr<scalar_t>(),
        B,
        L,
        D);
  });

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
