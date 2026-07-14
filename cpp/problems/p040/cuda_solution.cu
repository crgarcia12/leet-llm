#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

__global__ void append_kv_kernel(const float* residual, const float* wk, const float* wv,
                                 float* cache_k, float* cache_v, int context_before, int dim) {
  const int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= dim) return;
  float k = 0.0f, v = 0.0f;
  for (int in = 0; in < dim; ++in) {
    const float x = residual[in];
    k += wk[out * dim + in] * x;
    v += wv[out * dim + in] * x;
  }
  cache_k[context_before * dim + out] = k;
  cache_v[context_before * dim + out] = v;
}

__global__ void score_kernel(const float* query, const float* cache_k, float* scores,
                             int context, int dim) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= context) return;
  float dot = 0.0f;
  for (int i = 0; i < dim; ++i) dot += query[i] * cache_k[token * dim + i];
  scores[token] = dot / sqrtf(static_cast<float>(dim));
}

std::vector<float> decode_cpu(const std::vector<float>& query, std::vector<float>& cache_k,
                              std::vector<float>& cache_v, const std::vector<float>& wk,
                              const std::vector<float>& wv, const std::vector<float>& residual,
                              int context_before, int dim) {
  for (int out = 0; out < dim; ++out) {
    float k = 0.0f, v = 0.0f;
    for (int in = 0; in < dim; ++in) {
      k += wk[out * dim + in] * residual[in];
      v += wv[out * dim + in] * residual[in];
    }
    cache_k[context_before * dim + out] = k;
    cache_v[context_before * dim + out] = v;
  }
  const int context = context_before + 1;
  std::vector<float> scores(context);
  float maxv = -INFINITY;
  for (int t = 0; t < context; ++t) {
    float dot = 0.0f;
    for (int i = 0; i < dim; ++i) dot += query[i] * cache_k[t * dim + i];
    scores[t] = dot / std::sqrt(static_cast<float>(dim));
    maxv = std::max(maxv, scores[t]);
  }
  float denom = 0.0f;
  for (float& s : scores) {
    s = std::exp(s - maxv);
    denom += s;
  }
  for (float& s : scores) s /= denom;
  std::vector<float> output(dim, 0.0f);
  for (int t = 0; t < context; ++t)
    for (int i = 0; i < dim; ++i) output[i] += scores[t] * cache_v[t * dim + i];
  return output;
}

int main() {
  constexpr int dim = 4;
  const int context_before = 2;
  std::vector<float> cache_k((context_before + 1) * dim), cache_v((context_before + 1) * dim);
  std::vector<float> query{0.7f, -0.2f, 0.3f, 0.1f};
  std::vector<float> residual{0.5f, 1.0f, -0.25f, 0.75f};
  std::vector<float> wk(dim * dim, 0.0f), wv(dim * dim, 0.0f);
  for (int i = 0; i < dim; ++i) {
    wk[i * dim + i] = 1.0f;
    wv[i * dim + i] = 0.8f;
  }
  for (int i = 0; i < context_before * dim; ++i) {
    cache_k[i] = 0.1f * (i + 1);
    cache_v[i] = -0.05f * (i + 2);
  }

  std::vector<float> gpu_cache_k = cache_k, gpu_cache_v = cache_v;
  float *d_query = nullptr, *d_residual = nullptr, *d_wk = nullptr, *d_wv = nullptr;
  float *d_cache_k = nullptr, *d_cache_v = nullptr, *d_scores = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_residual, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wk, wk.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wv, wv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache_k, gpu_cache_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache_v, gpu_cache_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scores, (context_before + 1) * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_query, query.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_residual, residual.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), wk.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), wv.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cache_k, gpu_cache_k.data(), gpu_cache_k.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_cache_v, gpu_cache_v.data(), gpu_cache_v.size() * sizeof(float), cudaMemcpyHostToDevice));

  append_kv_kernel<<<1, 128>>>(d_residual, d_wk, d_wv, d_cache_k, d_cache_v, context_before, dim);
  score_kernel<<<1, 128>>>(d_query, d_cache_k, d_scores, context_before + 1, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> gpu_scores(context_before + 1);
  CUDA_CHECK(cudaMemcpy(gpu_cache_k.data(), d_cache_k, gpu_cache_k.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_cache_v.data(), d_cache_v, gpu_cache_v.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(gpu_scores.data(), d_scores, gpu_scores.size() * sizeof(float), cudaMemcpyDeviceToHost));

  auto cpu_output = decode_cpu(query, cache_k, cache_v, wk, wv, residual, context_before, dim);
  float maxv = -INFINITY;
  for (float s : gpu_scores) maxv = std::max(maxv, s);
  float denom = 0.0f;
  for (float& s : gpu_scores) {
    s = std::exp(s - maxv);
    denom += s;
  }
  for (float& s : gpu_scores) s /= denom;
  std::vector<float> gpu_output(dim, 0.0f);
  for (int t = 0; t < context_before + 1; ++t)
    for (int i = 0; i < dim; ++i) gpu_output[i] += gpu_scores[t] * gpu_cache_v[t * dim + i];

  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_wk));
  CUDA_CHECK(cudaFree(d_wv));
  CUDA_CHECK(cudaFree(d_cache_k));
  CUDA_CHECK(cudaFree(d_cache_v));
  CUDA_CHECK(cudaFree(d_scores));

  for (int i = 0; i < dim; ++i)
    if (std::abs(gpu_output[i] - cpu_output[i]) > 1e-5f) return 1;
  std::cout << "p040 autoregressive decode append/attend path validated\n";
  return 0;
}
