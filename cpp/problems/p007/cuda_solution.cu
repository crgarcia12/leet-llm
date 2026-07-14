#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

enum class Activation : int { ReLU = 0, GELUTanh = 1, SiLU = 2 };

__device__ float gelu_tanh(float x) {
  const float c = std::sqrt(2.0f / 3.14159265358979323846f);
  const float cubic = x * x * x;
  return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * cubic)));
}

__global__ void activation_kernel(const float* input, float* output, int count, int mode) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float x = input[index];
  if (mode == static_cast<int>(Activation::ReLU)) output[index] = fmaxf(0.0f, x);
  if (mode == static_cast<int>(Activation::GELUTanh)) output[index] = gelu_tanh(x);
  if (mode == static_cast<int>(Activation::SiLU)) output[index] = x / (1.0f + expf(-x));
}

float cpu_activation(float x, Activation activation) {
  if (activation == Activation::ReLU) return std::max(0.0f, x);
  if (activation == Activation::GELUTanh) {
    const float c = std::sqrt(2.0f / 3.14159265358979323846f);
    return 0.5f * x * (1.0f + std::tanh(c * (x + 0.044715f * x * x * x)));
  }
  return x / (1.0f + std::exp(-x));
}

bool run_case(const std::string& name, const std::vector<float>& input, Activation activation, float tolerance) {
  std::vector<float> expected(input.size()), actual(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) expected[i] = cpu_activation(input[i], activation);

  float *device_input = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

  activation_kernel<<<(static_cast<int>(input.size()) + 255) / 256, 256>>>(
      device_input, device_output, static_cast<int>(input.size()), static_cast<int>(activation));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, input.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  for (std::size_t i = 0; i < input.size(); ++i) {
    const float scale = std::max({1.0f, std::abs(expected[i]), std::abs(actual[i])});
    if (std::abs(expected[i] - actual[i]) > tolerance * scale) {
      std::cerr << name << " failed at index " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  const bool ok =
      run_case("ReLU signs and zero", {-3, -0.0f, 0.5f, 4}, Activation::ReLU, 2e-5f) &&
      run_case("GELU tanh approximation", {-3, -1, 0, 1, 3}, Activation::GELUTanh, 2e-5f) &&
      run_case("SiLU wide inputs", {-20, -2, 0, 2, 20}, Activation::SiLU, 2e-5f);

  if (!ok) return 1;
  std::cout << "p007 CUDA canonical solution passed activation validation\n";
  return 0;
}
