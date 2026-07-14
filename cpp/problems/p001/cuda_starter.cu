#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__global__ void dot_partial_kernel_todo(const float* lhs, const float* rhs, float* partials, int count) {
  __shared__ float scratch[kBlockSize];
  const int lane = threadIdx.x;
  // TODO(p001): load lhs[index] * rhs[index] for this lane and reduce to one partial per block.
  // This placeholder keeps the build working but is intentionally incorrect.
  scratch[lane] = 0.0f;
  __syncthreads();
  if (lane == 0) partials[blockIdx.x] = scratch[0];
}

float cpu_reference(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  double sum = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) sum += static_cast<double>(lhs[i]) * rhs[i];
  return static_cast<float>(sum);
}

float cuda_dot_todo(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.empty()) return 0.0f;
  float *device_lhs = nullptr, *device_rhs = nullptr, *device_partials = nullptr;
  const int count = static_cast<int>(lhs.size());
  const int blocks = (count + kBlockSize - 1) / kBlockSize;
  CUDA_CHECK(cudaMalloc(&device_lhs, lhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_rhs, rhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_partials, blocks * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_lhs, lhs.data(), lhs.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_rhs, rhs.data(), rhs.size() * sizeof(float), cudaMemcpyHostToDevice));
  dot_partial_kernel_todo<<<blocks, kBlockSize>>>(device_lhs, device_rhs, device_partials, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> partials(blocks);
  CUDA_CHECK(cudaMemcpy(partials.data(), device_partials, partials.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_lhs));
  CUDA_CHECK(cudaFree(device_rhs));
  CUDA_CHECK(cudaFree(device_partials));

  // TODO(p001): reduce block partials on GPU or CPU once kernel writes correct values.
  float sum = 0.0f;
  for (float v : partials) sum += v;
  return sum;
}

}  // namespace

int main() {
  const std::vector<float> lhs{1, -2, 3, -4};
  const std::vector<float> rhs{0.5f, 2.0f, -1.0f, -0.25f};
  const float expected = cpu_reference(lhs, rhs);
  const float actual = cuda_dot_todo(lhs, rhs);

  if (std::abs(expected - actual) <= 1e-5f) {
    std::cerr << "p001 starter unexpectedly passed; TODO implementation is incomplete\n";
    return 1;
  }
  std::cerr << "p001 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
