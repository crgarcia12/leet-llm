#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void softmax_todo_kernel(const float* logits, float* output, int rows, int columns) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = rows * columns;
  if (index < count) {
    // TODO(p009): row-wise max subtraction and denominator reduction.
    output[index] = expf(logits[index]);  // Incomplete and numerically unstable.
  }
}

}  // namespace

int main() {
  const std::vector<float> logits{10000, 10001, 9999};
  std::vector<float> output(logits.size(), 0.0f);

  float *device_logits = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_logits, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));

  softmax_todo_kernel<<<1, 64>>>(device_logits, device_output, 1, 3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_logits));
  CUDA_CHECK(cudaFree(device_output));

  const float sum = output[0] + output[1] + output[2];
  if (std::isfinite(sum) && std::abs(sum - 1.0f) <= 1e-4f) {
    std::cerr << "p009 starter unexpectedly passed; TODO stable softmax reductions are incomplete\n";
    return 1;
  }
  std::cerr << "p009 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
