#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void decoder_block_todo_kernel(const float* residual, float* output, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;

  // TODO(p035): implement pre-norm attention residual, then pre-norm MLP residual in order.
  output[idx] = residual[idx];
}

}  // namespace

int main() {
  float *d_residual = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_residual, 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, 2 * sizeof(float)));
  decoder_block_todo_kernel<<<1, 64>>>(d_residual, d_output, 2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_residual)); CUDA_CHECK(cudaFree(d_output));

  std::cerr << "p035 starter intentionally fails: TODO full decoder block ordering\n";
  return 1;
}
