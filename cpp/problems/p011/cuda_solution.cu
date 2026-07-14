#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>

namespace {

__global__ void residual_add_fp32_kernel(float* residual, const float* update, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) residual[index] += update[index];
}

__global__ void residual_add_fp16_roundtrip_kernel(float* residual, const float* update, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float sum = residual[index] + update[index];
    residual[index] = __half2float(__float2half_rn(sum));
  }
}

std::vector<float> cpu_reference(std::vector<float> residual,
                                 const std::vector<std::vector<float>>& updates,
                                 bool fp16_roundtrip) {
  for (const auto& update : updates) {
    for (std::size_t i = 0; i < residual.size(); ++i) {
      const float sum = residual[i] + update[i];
      residual[i] = fp16_roundtrip ? static_cast<float>(static_cast<__half>(sum)) : sum;
    }
  }
  return residual;
}

std::vector<float> cuda_accumulate(const std::vector<float>& initial,
                                   const std::vector<std::vector<float>>& updates,
                                   bool fp16_roundtrip) {
  std::vector<float> result(initial);
  float* d_residual = nullptr;
  float* d_update = nullptr;
  CUDA_CHECK(cudaMalloc(&d_residual, result.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_update, result.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_residual, result.data(), result.size() * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 128;
  const int blocks = static_cast<int>((result.size() + threads - 1) / threads);
  for (const auto& update : updates) {
    CUDA_CHECK(cudaMemcpy(d_update, update.data(), update.size() * sizeof(float), cudaMemcpyHostToDevice));
    if (fp16_roundtrip) {
      residual_add_fp16_roundtrip_kernel<<<blocks, threads>>>(d_residual, d_update, static_cast<int>(result.size()));
    } else {
      residual_add_fp32_kernel<<<blocks, threads>>>(d_residual, d_update, static_cast<int>(result.size()));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaMemcpy(result.data(), d_residual, result.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_update));
  return result;
}

bool run_case() {
  const std::vector<float> initial{4096.0f, -4096.0f};
  std::vector<std::vector<float>> updates(4, std::vector<float>{0.5f, -0.5f});

  const std::vector<float> cpu32 = cpu_reference(initial, updates, false);
  const std::vector<float> cpu16 = cpu_reference(initial, updates, true);
  const std::vector<float> gpu32 = cuda_accumulate(initial, updates, false);
  const std::vector<float> gpu16 = cuda_accumulate(initial, updates, true);

  for (int i = 0; i < 2; ++i) {
    if (std::abs(cpu32[i] - gpu32[i]) > 1e-6f || std::abs(cpu16[i] - gpu16[i]) > 1e-6f) {
      std::cerr << "policy mismatch at index " << i << '\n';
      return false;
    }
  }

  const float max_abs_diff = std::max(std::abs(gpu32[0] - gpu16[0]), std::abs(gpu32[1] - gpu16[1]));
  if (std::abs(gpu32[0] - 4098.0f) > 1e-6f || std::abs(gpu16[0] - 4096.0f) > 1e-6f ||
      std::abs(max_abs_diff - 2.0f) > 1e-6f) {
    std::cerr << "expected deterministic precision divergence was not reproduced\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!run_case()) return 1;
  std::cout << "p011 CUDA canonical solution passed residual precision-policy validation\n";
  return 0;
}
