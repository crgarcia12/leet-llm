#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void activation_todo_kernel(const float* input, float* output, int count, int mode) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    // TODO(p007): implement ReLU, GELU(tanh), and SiLU switch logic.
    output[index] = input[index];
  }
}

}  // namespace

int main() {
  const std::vector<float> input{-3, -1, 0, 1, 3};
  std::vector<float> output(input.size(), 0.0f);

  float *device_input = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

  activation_todo_kernel<<<1, 128>>>(device_input, device_output, static_cast<int>(input.size()), 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  const float expected = -0.158808f;
  if (std::abs(output[1] - expected) <= 1e-4f) {
    std::cerr << "p007 starter unexpectedly passed; TODO activation formulas are incomplete\n";
    return 1;
  }
  std::cerr << "p007 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
