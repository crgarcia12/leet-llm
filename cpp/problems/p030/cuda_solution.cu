#include "cuda_check.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__global__ void groupwise_quant_kernel(const float* weights, std::int8_t* q,
                                       float* scales, int out_channels,
                                       int in_channels, int group_size,
                                       int groups_per_row) {
  const int row = blockIdx.x;
  const int group = threadIdx.x;
  if (row >= out_channels || group >= groups_per_row) return;

  const int begin = group * group_size;
  const int end = min(in_channels, begin + group_size);
  float max_abs = 0.0f;
  for (int c = begin; c < end; ++c)
    max_abs = fmaxf(max_abs, fabsf(weights[row * in_channels + c]));

  const float scale = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
  scales[row * groups_per_row + group] = scale;
  for (int c = begin; c < end; ++c) {
    const float rounded = nearbyintf(weights[row * in_channels + c] / scale);
    q[row * in_channels + c] = static_cast<std::int8_t>(fminf(127.0f, fmaxf(-127.0f, rounded)));
  }
}

__global__ void groupwise_dequant_kernel(const std::int8_t* q, const float* scales,
                                         float* out, int out_channels,
                                         int in_channels, int group_size,
                                         int groups_per_row) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = out_channels * in_channels;
  if (index >= count) return;
  const int row = index / in_channels;
  const int col = index % in_channels;
  const int group = col / group_size;
  out[index] = static_cast<float>(q[index]) * scales[row * groups_per_row + group];
}

bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
  constexpr int out_channels = 2;
  constexpr int in_channels = 5;
  constexpr int group_size = 3;
  constexpr int groups_per_row = 2;
  constexpr int count = out_channels * in_channels;

  const std::vector<float> weights{
      -1.0f, -0.5f, 0.0f, 10.0f, 5.0f,
       0.1f, -0.1f, 0.05f, -0.02f, 0.0f,
  };

  float* d_w = nullptr;
  std::int8_t* d_q = nullptr;
  float *d_scales = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_w, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, count * sizeof(std::int8_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, out_channels * groups_per_row * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_w, weights.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  groupwise_quant_kernel<<<out_channels, groups_per_row>>>(
      d_w, d_q, d_scales, out_channels, in_channels, group_size, groups_per_row);
  CUDA_CHECK(cudaGetLastError());
  groupwise_dequant_kernel<<<1, 128>>>(
      d_q, d_scales, d_out, out_channels, in_channels, group_size, groups_per_row);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> scales(out_channels * groups_per_row);
  std::vector<float> dequantized(count);
  CUDA_CHECK(cudaMemcpy(scales.data(), d_scales, scales.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(dequantized.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_w)); CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_scales)); CUDA_CHECK(cudaFree(d_out));

  if (!near(scales[0], 1.0f / 127.0f) || !near(scales[1], 10.0f / 127.0f) ||
      !near(scales[2], 0.1f / 127.0f) || !near(scales[3], 0.02f / 127.0f)) return 1;
  if (!near(dequantized[0], -1.0f, 0.02f) || !near(dequantized[3], 10.0f, 0.05f) ||
      !near(dequantized[8], -0.02f, 0.01f)) return 1;

  const std::size_t bytes = count + out_channels * groups_per_row * sizeof(float);
  if (bytes != 26) return 1;

  std::cout << "p030 CUDA groupwise scales quantization passed\n";
  return 0;
}
