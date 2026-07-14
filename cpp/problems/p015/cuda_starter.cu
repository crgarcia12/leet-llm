#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void rope_todo_kernel(const float* input, float* output, int total_elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < total_elements) {
    // TODO(p015): rotate adjacent pairs by position-dependent angle.
    output[index] = input[index];
  }
}

}  // namespace

int main() {
  const std::vector<float> input{1.0f, 0.0f, 0.0f, 2.0f, 9.0f, 10.0f};
  std::vector<float> output(input.size(), 0.0f);

  float *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

  rope_todo_kernel<<<1, 64>>>(d_input, d_output, static_cast<int>(input.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));

  const bool unchanged = std::abs(output[0] - input[0]) <= 1e-6f && std::abs(output[1] - input[1]) <= 1e-6f;
  if (!unchanged) {
    std::cerr << "p015 starter should currently be a no-op placeholder\n";
    return 1;
  }
  std::cerr << "p015 starter intentionally fails validation until TODO rotation is implemented\n";
  return 1;
}
