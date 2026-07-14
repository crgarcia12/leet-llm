#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void dequantize_q4_todo_kernel(const unsigned char* packed, const float* scales,
                                          float* weights, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  // TODO(p032): unpack signed INT4 values and apply per-group scales.
  weights[idx] = 0.0f;
}

}  // namespace

int main() {
  unsigned char* d_packed = nullptr;
  float *d_scales = nullptr, *d_weights = nullptr;
  CUDA_CHECK(cudaMalloc(&d_packed, 5));
  CUDA_CHECK(cudaMalloc(&d_scales, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, 10 * sizeof(float)));
  dequantize_q4_todo_kernel<<<1, 64>>>(d_packed, d_scales, d_weights, 10);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_packed)); CUDA_CHECK(cudaFree(d_scales)); CUDA_CHECK(cudaFree(d_weights));

  std::cerr << "p032 starter intentionally fails: TODO staged dequantization and GEMV\n";
  return 1;
}
