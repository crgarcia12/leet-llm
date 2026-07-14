#include "cuda_check.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__global__ void fused_q4_gemv_kernel(const std::uint8_t* packed, const float* scales,
                                     const float* input, float* output,
                                     int out_channels, int in_channels,
                                     int group_size, int groups_per_row) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= out_channels) return;

  float sum = 0.0f;
  for (int col = 0; col < in_channels; ++col) {
    const int logical = row * in_channels + col;
    const std::uint8_t byte = packed[logical / 2];
    const std::uint8_t nibble = (logical % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
    const int q = nibble >= 8 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
    const float scale = scales[row * groups_per_row + col / group_size];
    sum += static_cast<float>(q) * scale * input[col];
  }
  output[row] = sum;
}

bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
  constexpr int out_channels = 2;
  constexpr int in_channels = 5;
  constexpr int group_size = 3;
  constexpr int groups_per_row = 2;

  const std::vector<std::uint8_t> packed{0xc8, 0x30, 0x17, 0x2f, 0x0e};
  const std::vector<float> scales{0.25f, 0.50f, 0.10f, 0.20f};
  const std::vector<float> input{1.0f, -2.0f, 0.5f, 1.5f, -1.0f};

  std::uint8_t* d_packed = nullptr;
  float *d_scales = nullptr, *d_input = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_packed, packed.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, scales.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_input, in_channels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, out_channels * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), in_channels * sizeof(float), cudaMemcpyHostToDevice));

  fused_q4_gemv_kernel<<<1, 64>>>(d_packed, d_scales, d_input, d_output,
                                  out_channels, in_channels, group_size, groups_per_row);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(out_channels);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, out_channels * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_packed)); CUDA_CHECK(cudaFree(d_scales));
  CUDA_CHECK(cudaFree(d_input)); CUDA_CHECK(cudaFree(d_output));

  if (!near(output[0], -1.25f) || !near(output[1], -0.2f, 2e-4f)) return 1;
  const std::size_t logical_weight_bytes = packed.size() + scales.size() * sizeof(float);
  if (logical_weight_bytes != 21) return 1;

  std::cout << "p033 CUDA fused Q4 GEMV passed\n";
  return 0;
}
