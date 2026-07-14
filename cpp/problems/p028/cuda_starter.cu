#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void quantize_kv_todo_kernel(const float* k_in, const float* v_in,
                                        signed char* k_q, signed char* v_q,
                                        float* shared_scales, int element_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= element_count) return;

  // TODO(p028): compute independent K and V per-vector scales and quantize each vector.
  k_q[idx] = static_cast<signed char>(k_in[idx]);
  v_q[idx] = static_cast<signed char>(v_in[idx]);
  shared_scales[idx] = 1.0f;
}

}  // namespace

int main() {
  float *d_k = nullptr, *d_v = nullptr;
  signed char *d_kq = nullptr, *d_vq = nullptr;
  float* d_scales = nullptr;
  CUDA_CHECK(cudaMalloc(&d_k, 8 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, 8 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kq, 8));
  CUDA_CHECK(cudaMalloc(&d_vq, 8));
  CUDA_CHECK(cudaMalloc(&d_scales, 8 * sizeof(float)));
  quantize_kv_todo_kernel<<<1, 64>>>(d_k, d_v, d_kq, d_vq, d_scales, 8);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_kq)); CUDA_CHECK(cudaFree(d_vq)); CUDA_CHECK(cudaFree(d_scales));

  std::cerr << "p028 starter intentionally fails: TODO independent K/V quantization metadata\n";
  return 1;
}
