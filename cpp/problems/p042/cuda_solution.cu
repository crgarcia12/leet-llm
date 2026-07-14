#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

__global__ void abs_diff_kernel(const float* reference, const float* candidate, float* diff, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) diff[i] = fabsf(reference[i] - candidate[i]);
}

float max_abs_diff_gpu(const std::vector<float>& a, const std::vector<float>& b) {
  float *d_a = nullptr, *d_b = nullptr, *d_diff = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, b.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_diff, a.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
  abs_diff_kernel<<<(static_cast<int>(a.size()) + 127) / 128, 128>>>(d_a, d_b, d_diff, static_cast<int>(a.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> diff(a.size());
  CUDA_CHECK(cudaMemcpy(diff.data(), d_diff, diff.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_diff));
  float maximum = 0.0f;
  for (float v : diff) maximum = std::max(maximum, v);
  return maximum;
}

int main() {
  const std::vector<std::string> names{
      "layer.0.attention_norm", "layer.0.rope.query", "layer.0.rope.key", "logits"};
  const std::vector<std::vector<float>> reference{
      {0.1f, 0.2f, 0.3f}, {0.4f, -0.5f, 0.6f}, {0.7f, 0.8f, -0.9f}, {1.0f, 0.5f, -0.25f}};
  auto candidate = reference;
  candidate[1][2] += 5e-3f;  // first intentional divergence.
  candidate[3][0] += 1e-3f;

  const float tolerance = 1e-3f;
  int first_divergent = -1;
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    const float max_diff = max_abs_diff_gpu(reference[i], candidate[i]);
    float cpu_max = 0.0f;
    for (int j = 0; j < static_cast<int>(reference[i].size()); ++j)
      cpu_max = std::max(cpu_max, std::fabs(reference[i][j] - candidate[i][j]));
    if (std::abs(max_diff - cpu_max) > 1e-6f) return 1;
    if (first_divergent < 0 && max_diff > tolerance) first_divergent = i;
  }

  if (first_divergent != 1 || names[first_divergent] != "layer.0.rope.query") return 1;
  std::cout << "p042 capture parity and first-divergence localization validated\n";
  return 0;
}
