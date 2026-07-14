#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void fused_q4_gemv_todo_kernel(const unsigned char* packed, const float* scales,
                                          const float* input, float* output,
                                          int out_channels) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= out_channels) return;

  // TODO(p033): fuse Q4 unpack + scale lookup + GEMV accumulation per output row.
  output[row] = 0.0f;
}

}  // namespace

int main() {
  unsigned char* d_packed = nullptr;
  float *d_scales = nullptr, *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_packed, 5));
  CUDA_CHECK(cudaMalloc(&d_scales, 4 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_in, 5 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, 2 * sizeof(float)));
  fused_q4_gemv_todo_kernel<<<1, 64>>>(d_packed, d_scales, d_in, d_out, 2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_packed)); CUDA_CHECK(cudaFree(d_scales));
  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));

  std::cerr << "p033 starter intentionally fails: TODO fused Q4 GEMV kernel\n";
  return 1;
}
