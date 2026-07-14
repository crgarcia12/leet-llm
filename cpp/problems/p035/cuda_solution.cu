#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void rmsnorm_kernel(const float* input, const float* gamma,
                               float* output, int width) {
  __shared__ float inv_rms;
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int i = 0; i < width; ++i) sum += input[i] * input[i];
    inv_rms = rsqrtf(sum / width + 1e-5f);
  }
  __syncthreads();
  const int idx = threadIdx.x;
  if (idx < width) output[idx] = input[idx] * inv_rms * gamma[idx];
}

__global__ void gemv_kernel(const float* matrix, const float* vector,
                            float* output, int rows, int cols) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int c = 0; c < cols; ++c) sum += matrix[row * cols + c] * vector[c];
  output[row] = sum;
}

__global__ void residual_add_kernel(const float* a, const float* b, float* out, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) out[idx] = a[idx] + b[idx];
}

__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    const float g = gate[idx];
    out[idx] = (g / (1.0f + expf(-g))) * up[idx];
  }
}

bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
  constexpr int d = 2;
  constexpr int f = 3;

  const std::vector<float> x{1.0f, 2.0f};
  const std::vector<float> attn_gamma{1.0f, 1.0f};
  const std::vector<float> mlp_gamma{1.0f, 1.0f};
  const std::vector<float> wo{1.0f, 0.0f, 0.0f, 1.0f};
  const std::vector<float> wgate{0.5f, -0.25f, 0.25f, 0.5f, -0.5f, 0.75f};
  const std::vector<float> wup{0.25f, 0.5f, -0.5f, 0.25f, 0.75f, -0.25f};
  const std::vector<float> wdown{0.5f, -0.25f, 0.25f, -0.5f, 0.75f, 0.5f};

  float *d_x = nullptr, *d_attn_norm = nullptr, *d_attn_proj = nullptr, *d_r1 = nullptr;
  float *d_mlp_norm = nullptr, *d_gate = nullptr, *d_up = nullptr, *d_hidden = nullptr, *d_down = nullptr, *d_r2 = nullptr;
  float *d_attn_gamma = nullptr, *d_mlp_gamma = nullptr;
  float *d_wo = nullptr, *d_wgate = nullptr, *d_wup = nullptr, *d_wdown = nullptr;

  CUDA_CHECK(cudaMalloc(&d_x, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_norm, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_proj, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r1, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mlp_norm, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gate, f * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up, f * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hidden, f * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_down, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_r2, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_gamma, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mlp_gamma, d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wo, d * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wgate, f * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wup, f * d * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wdown, d * f * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_x, x.data(), d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_attn_gamma, attn_gamma.data(), d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mlp_gamma, mlp_gamma.data(), d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wo, wo.data(), d * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wgate, wgate.data(), f * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wup, wup.data(), f * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wdown, wdown.data(), d * f * sizeof(float), cudaMemcpyHostToDevice));

  rmsnorm_kernel<<<1, 32>>>(d_x, d_attn_gamma, d_attn_norm, d);
  CUDA_CHECK(cudaGetLastError());
  gemv_kernel<<<1, 32>>>(d_wo, d_attn_norm, d_attn_proj, d, d);
  CUDA_CHECK(cudaGetLastError());
  residual_add_kernel<<<1, 32>>>(d_x, d_attn_proj, d_r1, d);
  CUDA_CHECK(cudaGetLastError());
  rmsnorm_kernel<<<1, 32>>>(d_r1, d_mlp_gamma, d_mlp_norm, d);
  CUDA_CHECK(cudaGetLastError());
  gemv_kernel<<<1, 32>>>(d_wgate, d_mlp_norm, d_gate, f, d);
  CUDA_CHECK(cudaGetLastError());
  gemv_kernel<<<1, 32>>>(d_wup, d_mlp_norm, d_up, f, d);
  CUDA_CHECK(cudaGetLastError());
  swiglu_kernel<<<1, 32>>>(d_gate, d_up, d_hidden, f);
  CUDA_CHECK(cudaGetLastError());
  gemv_kernel<<<1, 32>>>(d_wdown, d_hidden, d_down, d, f);
  CUDA_CHECK(cudaGetLastError());
  residual_add_kernel<<<1, 32>>>(d_r1, d_down, d_r2, d);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> r1(d), r2(d), attn_norm(d);
  CUDA_CHECK(cudaMemcpy(attn_norm.data(), d_attn_norm, d * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r1.data(), d_r1, d * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r2.data(), d_r2, d * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_attn_norm)); CUDA_CHECK(cudaFree(d_attn_proj));
  CUDA_CHECK(cudaFree(d_r1)); CUDA_CHECK(cudaFree(d_mlp_norm)); CUDA_CHECK(cudaFree(d_gate));
  CUDA_CHECK(cudaFree(d_up)); CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_down));
  CUDA_CHECK(cudaFree(d_r2)); CUDA_CHECK(cudaFree(d_attn_gamma)); CUDA_CHECK(cudaFree(d_mlp_gamma));
  CUDA_CHECK(cudaFree(d_wo)); CUDA_CHECK(cudaFree(d_wgate)); CUDA_CHECK(cudaFree(d_wup)); CUDA_CHECK(cudaFree(d_wdown));

  if (!near(r1[0], x[0] + attn_norm[0]) || !near(r1[1], x[1] + attn_norm[1])) return 1;
  if (!(std::isfinite(r2[0]) && std::isfinite(r2[1]))) return 1;

  std::cout << "p035 CUDA decoder block staging passed\n";
  return 0;
}
