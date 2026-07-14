#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void streaming_triad_todo_kernel(const float* x, float* y, float alpha, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    // TODO(p006): apply y = alpha * x + y and benchmark with CUDA events.
    y[index] += x[index];  // Incorrect placeholder, ignores alpha.
  }
}

}  // namespace

int main() {
  constexpr int count = 1024;
  std::vector<float> x(count, 2.0f), y(count, 1.0f);

  float *device_x = nullptr, *device_y = nullptr;
  CUDA_CHECK(cudaMalloc(&device_x, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_y, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_x, x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_y, y.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  streaming_triad_todo_kernel<<<(count + 255) / 256, 256>>>(device_x, device_y, 1.25f, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(y.data(), device_y, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_x));
  CUDA_CHECK(cudaFree(device_y));

  const float expected = 1.0f + 1.25f * 2.0f;
  if (std::abs(y[0] - expected) <= 1e-5f) {
    std::cerr << "p006 starter unexpectedly passed; TODO roofline kernel math is incomplete\n";
    return 1;
  }
  std::cerr << "p006 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
