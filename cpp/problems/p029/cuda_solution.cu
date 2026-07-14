#include "cuda_check.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__global__ void symmetric_int8_quant_kernel(const float* input, std::int8_t* q,
                                            float* scale_out, int count) {
  __shared__ float max_abs;
  if (threadIdx.x == 0) max_abs = 0.0f;
  __syncthreads();

  const int idx = threadIdx.x;
  if (idx < count) atomicMax(reinterpret_cast<int*>(&max_abs), __float_as_int(fabsf(input[idx])));
  __syncthreads();

  if (threadIdx.x == 0) scale_out[0] = max_abs == 0.0f ? 1.0f : max_abs / 127.0f;
  __syncthreads();

  if (idx < count) {
    const float scale = scale_out[0];
    const float rounded = nearbyintf(input[idx] / scale);
    const float clamped = fminf(127.0f, fmaxf(-127.0f, rounded));
    q[idx] = static_cast<std::int8_t>(clamped);
  }
}

__global__ void dequant_kernel(const std::int8_t* q, const float* scale,
                               float* output, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) output[idx] = static_cast<float>(q[idx]) * scale[0];
}

bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
  const std::vector<float> input{-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  constexpr int count = 5;

  float* d_in = nullptr;
  std::int8_t* d_q = nullptr;
  float *d_scale = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, count * sizeof(std::int8_t)));
  CUDA_CHECK(cudaMalloc(&d_scale, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_in, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  symmetric_int8_quant_kernel<<<1, 128>>>(d_in, d_q, d_scale, count);
  CUDA_CHECK(cudaGetLastError());
  dequant_kernel<<<1, 128>>>(d_q, d_scale, d_out, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::int8_t> q(count);
  std::vector<float> out(count);
  float scale = 0.0f;
  CUDA_CHECK(cudaMemcpy(q.data(), d_q, count * sizeof(std::int8_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&scale, d_scale, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_scale)); CUDA_CHECK(cudaFree(d_out));

  if (!near(scale, 2.0f / 127.0f, 1e-5f)) return 1;
  const std::vector<std::int8_t> expected_q{-127, -64, 0, 64, 127};
  if (q != expected_q) return 1;
  if (!near(out[1], -128.0f / 127.0f, 1e-4f)) return 1;

  std::cout << "p029 CUDA symmetric int8 quantization passed\n";
  return 0;
}
