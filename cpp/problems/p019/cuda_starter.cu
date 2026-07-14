#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void online_attention_todo_kernel(const float* scores,
                                             const float* values,
                                             float* output,
                                             int count) {
  // TODO(p019): track running max and rescale previous state when max increases.
  float denominator = 0.0f;
  float accumulator = 0.0f;
  for (int i = 0; i < count; ++i) {
    const float beta = expf(scores[i]);
    denominator += beta;
    accumulator += beta * values[i];
  }
  output[0] = accumulator / denominator;
}

}  // namespace

int main() {
  const std::vector<float> scores{1.0f, 3.0f};
  const std::vector<float> values{2.0f, 10.0f};
  std::vector<float> output(1, 0.0f);

  float *d_scores = nullptr, *d_values = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, scores.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_values, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_scores, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  online_attention_todo_kernel<<<1, 1>>>(d_scores, d_values, d_output, static_cast<int>(scores.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_output));

  const float expected = 9.046376f;
  if (std::abs(output[0] - expected) <= 1e-3f) {
    std::cerr << "p019 starter unexpectedly passed; TODO online rescaling is missing\n";
    return 1;
  }
  std::cerr << "p019 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
