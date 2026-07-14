#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void decode_weight_format_todo_kernel(const unsigned char* bytes,
                                                 unsigned int* version_out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  // TODO(p036): decode little-endian preamble fields and validate payload bounds.
  version_out[0] = bytes[8];
}

}  // namespace

int main() {
  unsigned char* d_bytes = nullptr;
  unsigned int* d_version = nullptr;
  CUDA_CHECK(cudaMalloc(&d_bytes, 32));
  CUDA_CHECK(cudaMalloc(&d_version, sizeof(unsigned int)));
  decode_weight_format_todo_kernel<<<1, 1>>>(d_bytes, d_version);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_bytes)); CUDA_CHECK(cudaFree(d_version));

  std::cerr << "p036 starter intentionally fails: TODO bounds-checked weight-format decode\n";
  return 1;
}
