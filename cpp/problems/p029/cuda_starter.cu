#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void symmetric_int8_todo_kernel(const float* input, signed char* q,
                                           float* scale_out, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  // TODO(p029): derive one tensor-wide scale and quantize to [-127, 127].
  scale_out[0] = 1.0f;
  q[idx] = 0;
}

}  // namespace

int main() {
  float *d_in = nullptr, *d_scale = nullptr;
  signed char* d_q = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, 5 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, 5));
  CUDA_CHECK(cudaMalloc(&d_scale, sizeof(float)));
  symmetric_int8_todo_kernel<<<1, 64>>>(d_in, d_q, d_scale, 5);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_q)); CUDA_CHECK(cudaFree(d_scale));

  std::cerr << "p029 starter intentionally fails: TODO symmetric INT8 quantization\n";
  return 1;
}
