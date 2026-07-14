#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void rmsnorm_todo_kernel(const float* input, const float* gamma, float* output,
                                    int rows, int width, float epsilon) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * width;
  if (index < count) {
    // TODO(p010): compute row RMS with epsilon and apply x / rms * gamma.
    output[index] = input[index] * gamma[index % width];
  }
}

}  // namespace

int main() {
  const std::vector<float> input{3, 4};
  const std::vector<float> gamma{2, 0.5f};
  std::vector<float> output(2, 0.0f);

  float *d_input = nullptr, *d_gamma = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice));

  rmsnorm_todo_kernel<<<1, 64>>>(d_input, d_gamma, d_output, 1, 2, 1e-6f);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_output));

  const std::vector<float> expected{1.697056f, 0.565685f};
  const bool passed = std::abs(output[0] - expected[0]) <= 1e-4f && std::abs(output[1] - expected[1]) <= 1e-4f;
  if (passed) {
    std::cerr << "p010 starter unexpectedly passed; TODO RMS reduction is incomplete\n";
    return 1;
  }
  std::cerr << "p010 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
