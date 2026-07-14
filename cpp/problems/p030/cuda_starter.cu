#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void groupwise_quant_todo_kernel(const float* weights, signed char* q,
                                            float* scales, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  // TODO(p030): compute per-row, per-group scales and quantize each group independently.
  q[idx] = 0;
  scales[0] = 1.0f;
}

}  // namespace

int main() {
  float* d_w = nullptr;
  signed char* d_q = nullptr;
  float* d_scales = nullptr;
  CUDA_CHECK(cudaMalloc(&d_w, 10 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, 10));
  CUDA_CHECK(cudaMalloc(&d_scales, 4 * sizeof(float)));
  groupwise_quant_todo_kernel<<<1, 64>>>(d_w, d_q, d_scales, 10);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_scales));

  std::cerr << "p030 starter intentionally fails: TODO groupwise scale indexing and tails\n";
  return 1;
}
