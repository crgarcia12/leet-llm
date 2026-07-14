#include "cuda_check.hpp"

#include <iostream>

namespace {

__global__ void paged_gather_todo_kernel(const float* physical_pages, const int* page_table,
                                         float* gathered, int page_size, int token_count) {
  const int logical_slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (logical_slot >= token_count) return;

  // TODO(p027): map logical page ordinals through page_table instead of assuming contiguous pages.
  gathered[logical_slot] = physical_pages[logical_slot];
}

}  // namespace

int main() {
  float *d_pages = nullptr, *d_out = nullptr;
  int* d_table = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pages, 6 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_table, 2 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_out, 4 * sizeof(float)));
  paged_gather_todo_kernel<<<1, 64>>>(d_pages, d_table, d_out, 2, 4);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_pages));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_out));

  std::cerr << "p027 starter intentionally fails: TODO paged address translation\n";
  return 1;
}
