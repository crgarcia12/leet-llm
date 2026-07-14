#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void p047_fused_qkv_rope_kernel(const float* residual, const float* weights,
                                            float* qkv, int width, int position) {
  const int projection = blockIdx.x;
  const int output = threadIdx.x;
  if (projection >= 3 || output >= width) return;
  float sum = 0;
  for (int input = 0; input < width; ++input)
    sum += weights[(projection * width + output) * width + input] * residual[input];
  qkv[projection * width + output] = sum;
  __syncthreads();
  if (projection < 2 && (output % 2) == 0 && output + 1 < width) {
    const float angle = position * powf(10000.0f, -static_cast<float>(output) / width);
    const float x = qkv[projection * width + output];
    const float y = qkv[projection * width + output + 1];
    qkv[projection * width + output] = x * cosf(angle) - y * sinf(angle);
    qkv[projection * width + output + 1] = x * sinf(angle) + y * cosf(angle);
  }
}

int main() {
  constexpr int width = 4, count = 3 * width;
  const std::vector<float> residual{1, 2, 3, 4};
  std::vector<float> weights(count * width), output(count), expected(count);
  for (int row = 0; row < count; ++row) weights[row * width + row % width] = 1;
  for (int projection = 0; projection < 3; ++projection)
    for (int i = 0; i < width; ++i) expected[projection * width + i] = residual[i];
  for (int projection = 0; projection < 2; ++projection)
    for (int i = 0; i < width; i += 2) {
      const float angle = 2 * std::pow(10000.0f, -static_cast<float>(i) / width);
      const float x = expected[projection * width + i], y = expected[projection * width + i + 1];
      expected[projection * width + i] = x * std::cos(angle) - y * std::sin(angle);
      expected[projection * width + i + 1] = x * std::sin(angle) + y * std::cos(angle);
    }
  float *device_residual = nullptr, *device_weights = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_residual, width * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_weights, weights.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_residual, residual.data(), width * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
  p047_fused_qkv_rope_kernel<<<3, width>>>(device_residual, device_weights, device_output, width, 2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_residual));
  CUDA_CHECK(cudaFree(device_weights));
  CUDA_CHECK(cudaFree(device_output));
  for (int i = 0; i < count; ++i)
    if (std::abs(output[i] - expected[i]) > 1e-5f + 1e-4f * std::abs(expected[i])) return 1;
  std::cout << "p047 fused QKV + RoPE matches CPU oracle\n";
}
