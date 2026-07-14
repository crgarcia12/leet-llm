#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void gemv_row_kernel_todo(const float* matrix, const float* vector, float* output,
                                     int rows, int columns) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    // TODO(p004): reduce across all columns for this output row.
    output[row] = columns > 0 ? matrix[row * columns] * vector[0] : 0.0f;
  }
}

}  // namespace

int main() {
  const std::vector<float> matrix{1, 2, 3, -1, 0.5f, 4};
  const std::vector<float> vector{2, -1, 0.5f};
  std::vector<float> output(2, 0.0f);

  float *device_matrix = nullptr, *device_vector = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_matrix, matrix.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_vector, vector.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_matrix, matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_vector, vector.data(), vector.size() * sizeof(float), cudaMemcpyHostToDevice));

  gemv_row_kernel_todo<<<1, 64>>>(device_matrix, device_vector, device_output, 2, 3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_matrix));
  CUDA_CHECK(cudaFree(device_vector));
  CUDA_CHECK(cudaFree(device_output));

  const std::vector<float> expected{1.5f, -0.5f};
  const bool passed = std::abs(output[0] - expected[0]) <= 1e-5f && std::abs(output[1] - expected[1]) <= 1e-5f;
  if (passed) {
    std::cerr << "p004 starter unexpectedly passed; TODO GEMV reduction is incomplete\n";
    return 1;
  }
  std::cerr << "p004 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
