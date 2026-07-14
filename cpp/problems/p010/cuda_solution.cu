#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__global__ void rmsnorm_kernel(const float* input, const float* gamma, float* output,
                               int rows, int width, float epsilon) {
  __shared__ float scratch[kBlockSize];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= rows) return;

  float partial = 0.0f;
  for (int col = lane; col < width; col += blockDim.x) {
    const float value = input[row * width + col];
    partial += value * value;
  }
  scratch[lane] = partial;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) scratch[lane] += scratch[lane + offset];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(width) + epsilon);

  for (int col = lane; col < width; col += blockDim.x)
    output[row * width + col] = input[row * width + col] * inv_rms * gamma[col];
}

std::vector<float> cpu_reference(const std::vector<float>& input, int rows, int width,
                                 const std::vector<float>& gamma, float epsilon) {
  std::vector<float> output(input.size(), 0.0f);
  for (int row = 0; row < rows; ++row) {
    double sum = 0.0;
    for (int col = 0; col < width; ++col) {
      const double v = input[row * width + col];
      sum += v * v;
    }
    const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum / width) + epsilon);
    for (int col = 0; col < width; ++col)
      output[row * width + col] = input[row * width + col] * inv_rms * gamma[col];
  }
  return output;
}

bool run_case(const std::string& name, const std::vector<float>& input, int rows, int width,
              const std::vector<float>& gamma, float epsilon, float tolerance) {
  if (rows == 0) return true;
  std::vector<float> expected = cpu_reference(input, rows, width, gamma, epsilon);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_input = nullptr, *d_gamma = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, actual.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice));

  rmsnorm_kernel<<<rows, kBlockSize>>>(d_input, d_gamma, d_output, rows, width, epsilon);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_output));

  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float bound = 4e-5f + tolerance * std::abs(expected[i]);
    if (std::abs(expected[i] - actual[i]) > bound) {
      std::cerr << name << " failed at index " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  const bool ok =
      run_case("mixed rows and scale", {1, 2, -3, 4, -2, 1}, 2, 3, {1, 0.5f, 2}, 1e-5f, 5e-5f) &&
      run_case("constant row", {4, 4, 4, 4}, 1, 4, {1, 1, 1, 1}, 1e-6f, 5e-5f) &&
      run_case("epsilon controls zero row", {0, 0, 0}, 1, 3, {2, 3, 4}, 0.25f, 5e-5f);

  if (!ok) return 1;
  std::cout << "p010 CUDA canonical solution passed RMSNorm validation\n";
  return 0;
}
