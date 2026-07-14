#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void gemm_todo_kernel(const float* lhs, const float* rhs, float* output,
                                 int m, int k, int n) {
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    // TODO(p005): cooperative tiled K-reduction with shared memory.
    output[row * n + col] = lhs[row * k] * rhs[col];
  }
}

}  // namespace

int main() {
  const std::vector<float> lhs{1, 2, 3, 4, 5, 6};
  const std::vector<float> rhs{1, 2, 0, -1, 3, 0};
  std::vector<float> output(4, 0.0f);

  float *device_lhs = nullptr, *device_rhs = nullptr, *device_out = nullptr;
  CUDA_CHECK(cudaMalloc(&device_lhs, lhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_rhs, rhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_out, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_lhs, lhs.data(), lhs.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_rhs, rhs.data(), rhs.size() * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threads(16, 16);
  dim3 blocks(1, 1);
  gemm_todo_kernel<<<blocks, threads>>>(device_lhs, device_rhs, device_out, 2, 3, 2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), device_out, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_lhs));
  CUDA_CHECK(cudaFree(device_rhs));
  CUDA_CHECK(cudaFree(device_out));

  const std::vector<float> expected{10, 0, 22, 3};
  bool pass = true;
  for (int i = 0; i < 4; ++i) pass = pass && std::abs(output[i] - expected[i]) <= 1e-5f;
  if (pass) {
    std::cerr << "p005 starter unexpectedly passed; TODO GEMM algorithm is incomplete\n";
    return 1;
  }
  std::cerr << "p005 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
