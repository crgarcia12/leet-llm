#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void residual_add_todo_kernel(float* residual, const float* update, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    // TODO(p011): add optional Float16 round trip after each add.
    residual[index] += update[index];
  }
}

}  // namespace

int main() {
  std::vector<float> residual{4096.0f};
  std::vector<float> update{0.5f};

  float *d_residual = nullptr, *d_update = nullptr;
  CUDA_CHECK(cudaMalloc(&d_residual, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_update, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_residual, residual.data(), sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_update, update.data(), sizeof(float), cudaMemcpyHostToDevice));

  for (int i = 0; i < 4; ++i) {
    residual_add_todo_kernel<<<1, 32>>>(d_residual, d_update, 1);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaMemcpy(residual.data(), d_residual, sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_update));

  // Float16-after-each-add policy should stay at 4096 here; placeholder returns 4098.
  if (std::abs(residual[0] - 4096.0f) <= 1e-6f) {
    std::cerr << "p011 starter unexpectedly passed; TODO precision policy branch is missing\n";
    return 1;
  }
  std::cerr << "p011 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
