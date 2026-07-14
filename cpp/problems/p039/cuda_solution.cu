#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void gather_embeddings_kernel(const int* token_ids, const float* table,
                                         float* residual, int sequence, int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= sequence * dim) return;
  const int token = index / dim;
  const int feature = index % dim;
  residual[index] = table[token_ids[token] * dim + feature];
}

__global__ void project_kv_kernel(const float* residual, const float* wk, const float* wv,
                                  float* keys, float* values, int sequence, int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= sequence * dim) return;
  const int token = index / dim;
  const int out = index % dim;
  float k = 0.0f;
  float v = 0.0f;
  for (int in = 0; in < dim; ++in) {
    const float x = residual[token * dim + in];
    k += wk[out * dim + in] * x;
    v += wv[out * dim + in] * x;
  }
  keys[index] = k;
  values[index] = v;
}

void prefill_cpu(const std::vector<int>& token_ids, const std::vector<float>& embedding,
                 const std::vector<float>& wk, const std::vector<float>& wv,
                 int dim, std::vector<float>& keys, std::vector<float>& values) {
  for (int t = 0; t < static_cast<int>(token_ids.size()); ++t) {
    for (int out = 0; out < dim; ++out) {
      float k = 0.0f, v = 0.0f;
      for (int in = 0; in < dim; ++in) {
        const float x = embedding[token_ids[t] * dim + in];
        k += wk[out * dim + in] * x;
        v += wv[out * dim + in] * x;
      }
      keys[t * dim + out] = k;
      values[t * dim + out] = v;
    }
  }
}

int main() {
  constexpr int vocab = 7;
  constexpr int dim = 4;
  const std::vector<int> prompt{1, 4, 2};

  std::vector<float> embedding(vocab * dim);
  for (int i = 0; i < vocab * dim; ++i) embedding[i] = ((i * 7) % 19 - 9) / 7.0f;
  std::vector<float> wk(dim * dim, 0.0f), wv(dim * dim, 0.0f);
  for (int i = 0; i < dim; ++i) {
    wk[i * dim + i] = 1.0f;
    wv[i * dim + i] = 0.5f + 0.1f * i;
  }

  int* d_tokens = nullptr;
  float *d_embedding = nullptr, *d_residual = nullptr, *d_wk = nullptr, *d_wv = nullptr;
  float *d_keys = nullptr, *d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tokens, prompt.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_embedding, embedding.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_residual, prompt.size() * dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wk, wk.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wv, wv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_keys, prompt.size() * dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_values, prompt.size() * dim * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_tokens, prompt.data(), prompt.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_embedding, embedding.data(), embedding.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), wk.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), wv.size() * sizeof(float), cudaMemcpyHostToDevice));

  const int count = static_cast<int>(prompt.size() * dim);
  gather_embeddings_kernel<<<(count + 127) / 128, 128>>>(d_tokens, d_embedding, d_residual,
                                                         static_cast<int>(prompt.size()), dim);
  project_kv_kernel<<<(count + 127) / 128, 128>>>(d_residual, d_wk, d_wv, d_keys, d_values,
                                                  static_cast<int>(prompt.size()), dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_keys(count), gpu_values(count), cpu_keys(count), cpu_values(count);
  CUDA_CHECK(cudaMemcpy(gpu_keys.data(), d_keys, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_values.data(), d_values, count * sizeof(float), cudaMemcpyDeviceToHost));

  prefill_cpu(prompt, embedding, wk, wv, dim, cpu_keys, cpu_values);
  CUDA_CHECK(cudaFree(d_tokens));
  CUDA_CHECK(cudaFree(d_embedding));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_wk));
  CUDA_CHECK(cudaFree(d_wv));
  CUDA_CHECK(cudaFree(d_keys));
  CUDA_CHECK(cudaFree(d_values));

  for (int i = 0; i < count; ++i) {
    if (std::abs(gpu_keys[i] - cpu_keys[i]) > 1e-5f ||
        std::abs(gpu_values[i] - cpu_values[i]) > 1e-5f) {
      return 1;
    }
  }
  std::cout << "p039 prompt prefill embedding gather and KV append validated\n";
  return 0;
}
