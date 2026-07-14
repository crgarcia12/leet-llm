#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__global__ void softmax_row_kernel(const float* logits, float* output, int rows, int columns) {
  __shared__ float scratch[kBlockSize];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= rows) return;

  float local_max = -CUDART_INF_F;
  for (int col = lane; col < columns; col += blockDim.x)
    local_max = fmaxf(local_max, logits[row * columns + col]);

  scratch[lane] = local_max;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) scratch[lane] = fmaxf(scratch[lane], scratch[lane + offset]);
    __syncthreads();
  }
  const float row_max = scratch[0];

  float local_sum = 0.0f;
  for (int col = lane; col < columns; col += blockDim.x)
    local_sum += expf(logits[row * columns + col] - row_max);
  scratch[lane] = local_sum;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) scratch[lane] += scratch[lane + offset];
    __syncthreads();
  }
  const float denom = scratch[0];

  for (int col = lane; col < columns; col += blockDim.x)
    output[row * columns + col] = expf(logits[row * columns + col] - row_max) / denom;
}

std::vector<float> cpu_reference(const std::vector<float>& logits, int rows, int columns) {
  std::vector<float> output(logits.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    float row_max = -std::numeric_limits<float>::infinity();
    for (int col = 0; col < columns; ++col)
      row_max = std::max(row_max, logits[row * columns + col]);
    double sum = 0.0;
    for (int col = 0; col < columns; ++col) {
      output[row * columns + col] = std::exp(logits[row * columns + col] - row_max);
      sum += output[row * columns + col];
    }
    for (int col = 0; col < columns; ++col)
      output[row * columns + col] = static_cast<float>(output[row * columns + col] / sum);
  }
  return output;
}

bool run_case(const std::string& name, const std::vector<float>& logits, int rows, int columns, float tolerance) {
  if (rows == 0) return true;
  std::vector<float> expected = cpu_reference(logits, rows, columns);
  std::vector<float> actual(expected.size(), 0.0f);

  float *device_logits = nullptr, *device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_logits, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_output, actual.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));

  softmax_row_kernel<<<rows, kBlockSize>>>(device_logits, device_output, rows, columns);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_logits));
  CUDA_CHECK(cudaFree(device_output));

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::abs(actual[i] - expected[i]) > tolerance) {
      std::cerr << name << " failed at index " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }

  for (int row = 0; row < rows; ++row) {
    float sum = 0.0f;
    for (int col = 0; col < columns; ++col) sum += actual[row * columns + col];
    if (std::abs(sum - 1.0f) > 3e-5f) {
      std::cerr << name << " row " << row << " does not sum to one: " << sum << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  std::vector<float> wide(257);
  for (int i = 0; i < 257; ++i) wide[i] = static_cast<float>((i % 29) - 14);

  const bool ok =
      run_case("ordinary rows", {1, 2, 3, -1, 0, 1}, 2, 3, 3e-5f) &&
      run_case("large positive logits", {10000, 10001, 9999}, 1, 3, 3e-5f) &&
      run_case("all-negative logits", {-10000, -10001, -9999}, 1, 3, 3e-5f) &&
      run_case("crosses threadgroup width", wide, 1, 257, 4e-5f);

  if (!ok) return 1;
  std::cout << "p009 CUDA canonical solution passed stable softmax validation\n";
  return 0;
}
