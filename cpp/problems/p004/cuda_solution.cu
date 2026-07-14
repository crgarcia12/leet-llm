#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__global__ void gemv_row_kernel(const float* matrix, const float* vector, float* output,
                                int rows, int columns) {
  __shared__ float scratch[kBlockSize];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  for (int col = lane; col < columns; col += blockDim.x)
    sum += matrix[row * columns + col] * vector[col];

  scratch[lane] = sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) scratch[lane] += scratch[lane + offset];
    __syncthreads();
  }
  if (lane == 0) output[row] = scratch[0];
}

std::vector<float> cpu_reference(const std::vector<float>& matrix, int rows, int columns,
                                 const std::vector<float>& vector) {
  std::vector<float> output(rows, 0.0f);
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < columns; ++col)
      sum += static_cast<double>(matrix[row * columns + col]) * vector[col];
    output[row] = static_cast<float>(sum);
  }
  return output;
}

bool run_case(const std::string& name, const std::vector<float>& matrix, int rows, int columns,
              const std::vector<float>& vector, float tolerance) {
  std::vector<float> expected = cpu_reference(matrix, rows, columns, vector);
  std::vector<float> actual(rows, 0.0f);
  if (rows == 0) return true;

  float *device_matrix = nullptr, *device_vector = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_matrix, matrix.size() * sizeof(float)));
  if (columns > 0) CUDA_CHECK(cudaMalloc(&device_vector, vector.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, rows * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_matrix, matrix.data(), matrix.size() * sizeof(float), cudaMemcpyHostToDevice));
  if (columns > 0)
    CUDA_CHECK(cudaMemcpy(device_vector, vector.data(), vector.size() * sizeof(float), cudaMemcpyHostToDevice));

  if (columns == 0) {
    CUDA_CHECK(cudaMemset(device_output, 0, rows * sizeof(float)));
  } else {
    gemv_row_kernel<<<rows, kBlockSize>>>(device_matrix, device_vector, device_output, rows, columns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, rows * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_matrix));
  if (device_vector) CUDA_CHECK(cudaFree(device_vector));
  CUDA_CHECK(cudaFree(device_output));

  for (int i = 0; i < rows; ++i) {
    const float scale = std::max({1.0f, std::abs(expected[i]), std::abs(actual[i])});
    if (std::abs(expected[i] - actual[i]) > tolerance * scale) {
      std::cerr << name << " failed at row " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  std::vector<float> wide_matrix(3 * 257), wide_vector(257);
  for (int i = 0; i < static_cast<int>(wide_matrix.size()); ++i)
    wide_matrix[i] = static_cast<float>((i % 29) - 14) / 13.0f;
  for (int i = 0; i < static_cast<int>(wide_vector.size()); ++i)
    wide_vector[i] = static_cast<float>((i % 17) - 8) / 9.0f;

  const bool ok =
      run_case("small projection", {1, 2, 3, -1, 0.5f, 4}, 2, 3, {2, -1, 0.5f}, 2e-5f) &&
      run_case("zero inner dimension", {}, 3, 0, {}, 2e-5f) &&
      run_case("crosses reduction boundary", wide_matrix, 3, 257, wide_vector, 2e-4f);

  if (!ok) return 1;
  std::cout << "p004 CUDA canonical solution passed GEMV validation\n";
  return 0;
}
