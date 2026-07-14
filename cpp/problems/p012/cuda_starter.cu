#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void fused_rmsnorm_gemv_todo_kernel(const float* input, const float* gamma,
                                               const float* weights, float* output,
                                               int width, int output_rows, float epsilon) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < output_rows) {
    // TODO(p012): compute RMS reduction then normalized weighted dot.
    float sum = 0.0f;
    for (int col = 0; col < width; ++col)
      sum += weights[row * width + col] * input[col];
    output[row] = sum;  // Missing normalization and gamma.
  }
}

}  // namespace

int main() {
  const std::vector<float> input{3, 4};
  const std::vector<float> gamma{2, 0.5f};
  const std::vector<float> weights{1, 2, -1, 1};
  std::vector<float> output(2, 0.0f);

  float *d_input = nullptr, *d_gamma = nullptr, *d_weights = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, weights.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));

  fused_rmsnorm_gemv_todo_kernel<<<1, 64>>>(d_input, d_gamma, d_weights, d_output, 2, 2, 1e-6f);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_output));

  const std::vector<float> expected{2.828426f, -1.131371f};
  const bool passed = std::abs(output[0] - expected[0]) <= 1e-3f && std::abs(output[1] - expected[1]) <= 1e-3f;
  if (passed) {
    std::cerr << "p012 starter unexpectedly passed; TODO fused normalization is incomplete\n";
    return 1;
  }
  std::cerr << "p012 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
