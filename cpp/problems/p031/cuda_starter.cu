#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void pack_int4_todo_kernel(const signed char* input, unsigned char* packed,
                                      int count) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int idx = pair * 2;
  if (idx >= count) return;

  // TODO(p031): pack low nibble first with two's-complement signed q4 values.
  packed[pair] = static_cast<unsigned char>(input[idx]);
}

}  // namespace

int main() {
  signed char* d_values = nullptr;
  unsigned char* d_packed = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, 7));
  CUDA_CHECK(cudaMalloc(&d_packed, 4));
  pack_int4_todo_kernel<<<1, 64>>>(d_values, d_packed, 7);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_packed));

  std::cerr << "p031 starter intentionally fails: TODO low-nibble-first INT4 packing\n";
  return 1;
}
