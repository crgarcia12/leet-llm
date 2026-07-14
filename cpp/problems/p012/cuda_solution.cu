#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

__global__ void fused_rmsnorm_gemv_kernel(const float* input, const float* gamma,
                                          const float* weights, float* output,
                                          int width, int output_rows, float epsilon) {
  __shared__ float scratch[kBlockSize];
  const int row = blockIdx.x;
  const int lane = threadIdx.x;
  if (row >= output_rows) return;

  float sum_sq = 0.0f;
  for (int col = lane; col < width; col += blockDim.x) {
    const float v = input[col];
    sum_sq += v * v;
  }
  scratch[lane] = sum_sq;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) scratch[lane] += scratch[lane + offset];
    __syncthreads();
  }
  const float inv_rms = rsqrtf(scratch[0] / static_cast<float>(width) + epsilon);

  float dot = 0.0f;
  for (int col = lane; col < width; col += blockDim.x) {
    const float normalized = input[col] * inv_rms * gamma[col];
    dot += weights[row * width + col] * normalized;
  }

  scratch[lane] = dot;
  __syncthreads();
  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset) scratch[lane] += scratch[lane + offset];
    __syncthreads();
  }
  if (lane == 0) output[row] = scratch[0];
}

std::vector<float> cpu_reference(const std::vector<float>& input,
                                 const std::vector<float>& gamma,
                                 const std::vector<float>& weights,
                                 int output_rows, int width, float epsilon) {
  double sum_sq = 0.0;
  for (float value : input) sum_sq += static_cast<double>(value) * value;
  const float inv_rms = 1.0f / std::sqrt(static_cast<float>(sum_sq / width) + epsilon);

  std::vector<float> out(output_rows, 0.0f);
  for (int row = 0; row < output_rows; ++row) {
    double dot = 0.0;
    for (int col = 0; col < width; ++col) {
      const float normalized = input[col] * inv_rms * gamma[col];
      dot += static_cast<double>(weights[row * width + col]) * normalized;
    }
    out[row] = static_cast<float>(dot);
  }
  return out;
}

bool run_case(const std::string& name,
              const std::vector<float>& input,
              const std::vector<float>& gamma,
              const std::vector<float>& weights,
              int output_rows,
              int width,
              float epsilon,
              float tolerance) {
  if (output_rows == 0) return true;

  std::vector<float> expected = cpu_reference(input, gamma, weights, output_rows, width, epsilon);
  std::vector<float> actual(output_rows, 0.0f);

  float *d_input = nullptr, *d_gamma = nullptr, *d_weights = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, weights.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output_rows * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));

  fused_rmsnorm_gemv_kernel<<<output_rows, kBlockSize>>>(d_input, d_gamma, d_weights, d_output,
                                                         width, output_rows, epsilon);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_output, output_rows * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_output));

  for (int i = 0; i < output_rows; ++i) {
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
  std::vector<float> wide_input(257), wide_gamma(257), wide_weights(3 * 257);
  for (int i = 0; i < 257; ++i) {
    wide_input[i] = static_cast<float>((i % 23) - 11) / 7.0f;
    wide_gamma[i] = 0.5f + static_cast<float>(i % 9) / 10.0f;
  }
  for (int i = 0; i < static_cast<int>(wide_weights.size()); ++i)
    wide_weights[i] = static_cast<float>((i % 31) - 15) / 19.0f;

  const bool ok =
      run_case("small fused projection", {1, -2, 3}, {1, 0.5f, 2}, {1, 2, 0, -1, 0.5f, 3}, 2, 3, 1e-5f, 8e-5f) &&
      run_case("crosses reduction boundary", wide_input, wide_gamma, wide_weights, 3, 257, 1e-6f, 1e-4f);

  if (!ok) return 1;
  std::cout << "p012 CUDA canonical solution passed fused RMSNorm+GEMV validation\n";
  return 0;
}
