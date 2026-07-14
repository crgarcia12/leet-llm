#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void q4_path_todo_kernel(const float* state, const unsigned char* packed,
                                    const float* scales, float* next,
                                    int dim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= dim) return;

  // TODO(p034): decode canonical low-nibble-first Q4 values and compare against float path.
  next[row] = state[row];
}

}  // namespace

int main() {
  float *d_state = nullptr, *d_scales = nullptr, *d_next = nullptr;
  unsigned char* d_packed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_state, 5 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_packed, 13));
  CUDA_CHECK(cudaMalloc(&d_scales, 10 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_next, 5 * sizeof(float)));
  q4_path_todo_kernel<<<1, 64>>>(d_state, d_packed, d_scales, d_next, 5);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_state)); CUDA_CHECK(cudaFree(d_packed));
  CUDA_CHECK(cudaFree(d_scales)); CUDA_CHECK(cudaFree(d_next));

  std::cerr << "p034 starter intentionally fails: TODO propagation capture + mismatch diagnosis\n";
  return 1;
}
