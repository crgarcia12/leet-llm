#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

__global__ void p009_stable_softmax_kernel(const float* input, float* output,
                                            int rows, int columns) {
  const int row = blockIdx.x;
  if (row >= rows || threadIdx.x != 0) return;
  float maximum = input[row * columns];
  for (int column = 1; column < columns; ++column)
    maximum = fmaxf(maximum, input[row * columns + column]);
  float sum = 0;
  for (int column = 0; column < columns; ++column) {
    const float value = expf(input[row * columns + column] - maximum);
    output[row * columns + column] = value;
    sum += value;
  }
  for (int column = 0; column < columns; ++column)
    output[row * columns + column] /= sum;
}

std::vector<float> cuda_softmax(const std::vector<float>& input, int rows, int columns) {
  if (rows < 0 || columns <= 0 || input.size() != static_cast<std::size_t>(rows * columns))
    throw std::invalid_argument("softmax requires rank-2 [rows, positive columns]");
  for (float value : input)
    if (!std::isfinite(value)) throw std::invalid_argument("softmax logits must be finite");
  if (rows == 0) return {};
  float *device_input = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float),
                        cudaMemcpyHostToDevice));
  p009_stable_softmax_kernel<<<rows, 32>>>(device_input, device_output, rows, columns);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> output(input.size());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_input));
  CUDA_CHECK(cudaFree(device_output));
  return output;
}

int main() {
  std::vector<float> logits(257);
  for (int i = 0; i < 257; ++i) logits[i] = 10000.0f + static_cast<float>(i - 128);
  const auto output = cuda_softmax(logits, 1, 257);
  double sum = 0;
  for (float value : output) sum += value;
  if (std::abs(sum - 1.0) > 3e-5 || !cuda_softmax({}, 0, 257).empty()) return 1;
  std::cout << "p009 stable-softmax CUDA canonical cases passed\n";
}
