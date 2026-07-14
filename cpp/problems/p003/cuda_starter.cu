#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void transpose_todo_kernel(const float* input, float* output, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    // TODO(p003): implement tiled transpose with [16][17] threadgroup/shared memory.
    output[idx] = input[idx];  // Incorrect placeholder copy.
  }
}

}  // namespace

int main() {
  const std::vector<float> input{1, 2, 3, 4, 5, 6};
  std::vector<float> output(input.size(), 0.0f);

  float *device_input = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));

  transpose_todo_kernel<<<1, 64>>>(device_input, device_output, static_cast<int>(input.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));

  const std::vector<float> expected_transpose{1, 4, 2, 5, 3, 6};
  bool matches = true;
  for (std::size_t i = 0; i < output.size(); ++i) matches = matches && std::abs(output[i] - expected_transpose[i]) <= 1e-6f;
  if (matches) {
    std::cerr << "p003 starter unexpectedly passed; TODO transpose implementation is missing\n";
    return 1;
  }
  std::cerr << "p003 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
