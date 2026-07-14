#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void p009_stable_softmax_starter(const float* input, float* output,
                                             int rows, int columns) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  float maximum = input[row * columns];
  for (int column = 1; column < columns; ++column)
    maximum = fmaxf(maximum, input[row * columns + column]);
  float denominator = 0;
  for (int column = 0; column < columns; ++column)
    denominator += expf(input[row * columns + column] - maximum);
  for (int column = 0; column < columns; ++column)
    output[row * columns + column] =
        expf(input[row * columns + column] - maximum) / denominator;
}

int main() {
  const std::vector<float> input{10000, 10001, 9999};
  std::vector<float> output(3);
  float *device_input = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, 3 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), 3 * sizeof(float), cudaMemcpyHostToDevice));
  p009_stable_softmax_starter<<<1, 32>>>(device_input, device_output, 1, 3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, 3 * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  if (std::abs(output[1] - 0.665241f) > 1e-5f) return 1;
  std::cout << "p009 CUDA starter stable-softmax check passed\n";
}
