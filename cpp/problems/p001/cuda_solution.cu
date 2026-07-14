#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__global__ void dot_partial_kernel(const float* lhs, const float* rhs, float* partials, int count) {
  __shared__ float scratch[kBlockSize];
  const int lane = threadIdx.x;
  const int stride = blockDim.x * gridDim.x;
  int index = blockIdx.x * blockDim.x + lane;
  float sum = 0.0f;
  while (index < count) {
    sum += lhs[index] * rhs[index];
    index += stride;
  }
  scratch[lane] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) scratch[lane] += scratch[lane + offset];
    __syncthreads();
  }
  if (lane == 0) partials[blockIdx.x] = scratch[0];
}

float cpu_reference(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  double sum = 0.0;
  for (std::size_t i = 0; i < lhs.size(); ++i) sum += static_cast<double>(lhs[i]) * rhs[i];
  return static_cast<float>(sum);
}

float cuda_dot(const std::vector<float>& lhs, const std::vector<float>& rhs) {
  if (lhs.empty()) return 0.0f;

  float *device_lhs = nullptr, *device_rhs = nullptr, *device_partials = nullptr;
  const int count = static_cast<int>(lhs.size());
  const int blocks = std::min(64, (count + kBlockSize - 1) / kBlockSize);

  CUDA_CHECK(cudaMalloc(&device_lhs, lhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_rhs, rhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_partials, blocks * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_lhs, lhs.data(), lhs.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_rhs, rhs.data(), rhs.size() * sizeof(float), cudaMemcpyHostToDevice));

  dot_partial_kernel<<<blocks, kBlockSize>>>(device_lhs, device_rhs, device_partials, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> partials(blocks);
  CUDA_CHECK(cudaMemcpy(partials.data(), device_partials, partials.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_lhs));
  CUDA_CHECK(cudaFree(device_rhs));
  CUDA_CHECK(cudaFree(device_partials));

  return std::accumulate(partials.begin(), partials.end(), 0.0f);
}

bool run_case(const std::string& name, const std::vector<float>& lhs, const std::vector<float>& rhs, float tolerance) {
  const float expected = cpu_reference(lhs, rhs);
  const float actual = cuda_dot(lhs, rhs);
  const float scale = std::max({1.0f, std::abs(expected), std::abs(actual)});
  if (std::abs(expected - actual) > tolerance * scale) {
    std::cerr << name << " failed: expected " << expected << ", got " << actual << '\n';
    return false;
  }
  return true;
}

}  // namespace

int main() {
  std::vector<float> long_lhs(1025), long_rhs(1025);
  for (int i = 0; i < 1025; ++i) {
    long_lhs[i] = static_cast<float>((i % 17) - 8) / 8.0f;
    long_rhs[i] = static_cast<float>((i % 11) - 5) / 5.0f;
  }

  const bool ok =
      run_case("empty vectors", {}, {}, 1e-5f) &&
      run_case("single element", {3.0f}, {-2.0f}, 1e-5f) &&
      run_case("mixed signs", {1, -2, 3, -4}, {0.5f, 2.0f, -1.0f, -0.25f}, 1e-5f) &&
      run_case("crosses threadgroup boundaries", long_lhs, long_rhs, 1e-4f);

  if (!ok) return 1;
  std::cout << "p001 CUDA canonical solution passed CPU reference checks\n";
  return 0;
}
