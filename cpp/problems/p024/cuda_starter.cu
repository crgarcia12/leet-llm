#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void wrong_head_major_todo_kernel(const float* logical,
                                             float* head_major,
                                             int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    // TODO(p024): use [L,H,T,D] indexing, not token-major indexing.
    head_major[index] = logical[index];
  }
}

}  // namespace

int main() {
  constexpr int count = 48;
  std::vector<float> logical(count, 0.0f);
  std::vector<float> head_major(count, -1.0f);
  for (int i = 0; i < count; ++i) logical[i] = static_cast<float>(i);

  float *d_logical = nullptr, *d_head = nullptr;
  CUDA_CHECK(cudaMalloc(&d_logical, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_head, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_logical, logical.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  wrong_head_major_todo_kernel<<<1, 128>>>(d_logical, d_head, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(head_major.data(), d_head, count * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_logical));
  CUDA_CHECK(cudaFree(d_head));

  const bool passed = std::abs(head_major[44] - 44.0f) <= 1e-6f;
  if (passed) {
    std::cerr << "p024 starter unexpectedly passed; TODO layout conversion is missing\n";
    return 1;
  }
  std::cerr << "p024 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
