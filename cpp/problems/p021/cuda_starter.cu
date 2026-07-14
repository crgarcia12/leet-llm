#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void sliding_window_todo_kernel(const float* values,
                                           float* output,
                                           int key_length,
                                           int head_dim) {
  // TODO(p021): respect inclusive lower window bound.
  for (int d = 0; d < head_dim; ++d) {
    float sum = 0.0f;
    for (int k = 0; k < key_length; ++k) sum += values[k * head_dim + d];
    output[d] = sum / static_cast<float>(key_length);
  }
}

}  // namespace

int main() {
  const std::vector<float> values{3.0f, 9.0f};
  std::vector<float> output(1, 0.0f);

  float *d_values = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  sliding_window_todo_kernel<<<1, 1>>>(d_values, d_output, 2, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_output));

  const float expected = 9.0f;  // W=1 must pick most recent value only.
  if (std::abs(output[0] - expected) <= 1e-5f) {
    std::cerr << "p021 starter unexpectedly passed; TODO window bound handling is missing\n";
    return 1;
  }
  std::cerr << "p021 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
