#include "cuda_check.hpp"
#include <cmath>
#include <iostream>
#include <vector>

__global__ void p005_lesson_kernel(const float* input, float* output, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) output[index] = input[index] * 5.0f + 1.0f;
}

int main() {
  constexpr int count = 257;
  std::vector<float> input(count), output(count);
  for (int i = 0; i < count; ++i) input[i] = static_cast<float>(i % 11 - 5);
  float *device_input = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  p005_lesson_kernel<<<(count + 127) / 128, 128>>>(device_input, device_output, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  for (int i = 0; i < count; ++i)
    if (std::abs(output[i] - (input[i] * 5.0f + 1.0f)) > 1e-5f) return 1;
  std::cout << "p005 CUDA canonical kernel passed CPU oracle comparison\n";
}
