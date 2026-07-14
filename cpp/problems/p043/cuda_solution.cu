#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void fused_rmsnorm_qkv_kernel(const float* input, const float* gamma,
                                         const float* wq, const float* wk, const float* wv,
                                         float* fused, int sequence, int model_dim,
                                         int query_dim, int kv_dim, float epsilon) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = sequence * (query_dim + 2 * kv_dim);
  if (idx >= total) return;
  const int token = idx / (query_dim + 2 * kv_dim);
  const int out = idx % (query_dim + 2 * kv_dim);

  float mean_square = 0.0f;
  for (int i = 0; i < model_dim; ++i) {
    const float x = input[token * model_dim + i];
    mean_square += x * x;
  }
  const float inv_rms = rsqrtf(mean_square / model_dim + epsilon);
  const float* weights = out < query_dim ? wq : (out < query_dim + kv_dim ? wk : wv);
  const int row = out < query_dim ? out : (out < query_dim + kv_dim ? out - query_dim : out - query_dim - kv_dim);
  float sum = 0.0f;
  for (int i = 0; i < model_dim; ++i) {
    const float normalized = input[token * model_dim + i] * inv_rms * gamma[i];
    sum += weights[row * model_dim + i] * normalized;
  }
  fused[idx] = sum;
}

void fused_cpu(const std::vector<float>& input, const std::vector<float>& gamma,
               const std::vector<float>& wq, const std::vector<float>& wk, const std::vector<float>& wv,
               int sequence, int model_dim, int query_dim, int kv_dim, float epsilon,
               std::vector<float>& out) {
  const int width = query_dim + 2 * kv_dim;
  for (int t = 0; t < sequence; ++t) {
    float mean_square = 0.0f;
    for (int i = 0; i < model_dim; ++i) {
      const float x = input[t * model_dim + i];
      mean_square += x * x;
    }
    const float inv_rms = 1.0f / std::sqrt(mean_square / model_dim + epsilon);
    for (int o = 0; o < width; ++o) {
      const std::vector<float>& weights = o < query_dim ? wq : (o < query_dim + kv_dim ? wk : wv);
      const int row = o < query_dim ? o : (o < query_dim + kv_dim ? o - query_dim : o - query_dim - kv_dim);
      float sum = 0.0f;
      for (int i = 0; i < model_dim; ++i) {
        sum += weights[row * model_dim + i] * input[t * model_dim + i] * inv_rms * gamma[i];
      }
      out[t * width + o] = sum;
    }
  }
}

int main() {
  constexpr int sequence = 3;
  constexpr int model_dim = 4;
  constexpr int query_dim = 4;
  constexpr int kv_dim = 2;
  constexpr float epsilon = 1e-5f;

  std::vector<float> input(sequence * model_dim), gamma(model_dim), wq(query_dim * model_dim),
      wk(kv_dim * model_dim), wv(kv_dim * model_dim);
  for (int i = 0; i < static_cast<int>(input.size()); ++i) input[i] = ((i * 5) % 17 - 8) / 9.0f;
  for (int i = 0; i < model_dim; ++i) gamma[i] = 0.75f + 0.1f * i;
  for (int i = 0; i < static_cast<int>(wq.size()); ++i) wq[i] = ((i * 3) % 11 - 5) / 7.0f;
  for (int i = 0; i < static_cast<int>(wk.size()); ++i) wk[i] = ((i * 7) % 13 - 6) / 8.0f;
  for (int i = 0; i < static_cast<int>(wv.size()); ++i) wv[i] = ((i * 11) % 19 - 9) / 10.0f;

  const int fused_width = query_dim + 2 * kv_dim;
  std::vector<float> gpu(sequence * fused_width), cpu(sequence * fused_width);

  float *d_input = nullptr, *d_gamma = nullptr, *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr, *d_fused = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gamma, gamma.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wq, wq.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wk, wk.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wv, wv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_fused, gpu.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wq, wq.data(), wq.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), wk.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), wv.size() * sizeof(float), cudaMemcpyHostToDevice));

  fused_rmsnorm_qkv_kernel<<<(static_cast<int>(gpu.size()) + 127) / 128, 128>>>(
      d_input, d_gamma, d_wq, d_wk, d_wv, d_fused,
      sequence, model_dim, query_dim, kv_dim, epsilon);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(gpu.data(), d_fused, gpu.size() * sizeof(float), cudaMemcpyDeviceToHost));

  fused_cpu(input, gamma, wq, wk, wv, sequence, model_dim, query_dim, kv_dim, epsilon, cpu);

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gamma));
  CUDA_CHECK(cudaFree(d_wq));
  CUDA_CHECK(cudaFree(d_wk));
  CUDA_CHECK(cudaFree(d_wv));
  CUDA_CHECK(cudaFree(d_fused));

  for (int i = 0; i < static_cast<int>(gpu.size()); ++i)
    if (std::abs(gpu[i] - cpu[i]) > 5e-5f + 1e-4f * std::abs(cpu[i])) return 1;
  std::cout << "p043 fused RMSNorm+QKV projection validated\n";
  return 0;
}
