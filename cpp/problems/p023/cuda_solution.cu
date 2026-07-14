#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void append_kv_kernel(float* cache_k, float* cache_v,
                                 const float* token_k, const float* token_v,
                                 int capacity, int kv_heads, int head_dim,
                                 int slot) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = kv_heads * head_dim;
  if (index >= count) return;
  const int cache_index = slot * count + index;
  cache_k[cache_index] = token_k[index];
  cache_v[cache_index] = token_v[index];
}

__global__ void cached_single_token_attention_kernel(const float* query,
                                                     const float* cache_k,
                                                     const float* cache_v,
                                                     float* output,
                                                     int token_count,
                                                     int query_heads,
                                                     int kv_heads,
                                                     int head_dim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= query_heads) return;

  const int q_head = row;
  const int group_size = query_heads / kv_heads;
  const int kv_head = q_head / group_size;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float maximum = -CUDART_INF_F;
  float denominator = 0.0f;
  constexpr int kMaxHeadDim = 8;
  float accumulator[kMaxHeadDim] = {0.0f};

  for (int t = 0; t < token_count; ++t) {
    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d)
      score += query[q_head * head_dim + d] *
               cache_k[(t * kv_heads + kv_head) * head_dim + d];
    score *= scale;

    const float next_maximum = fmaxf(maximum, score);
    const float alpha = (maximum == -CUDART_INF_F) ? 0.0f : expf(maximum - next_maximum);
    const float beta = expf(score - next_maximum);

    denominator = denominator * alpha + beta;
    for (int d = 0; d < head_dim; ++d)
      accumulator[d] = accumulator[d] * alpha + beta * cache_v[(t * kv_heads + kv_head) * head_dim + d];
    maximum = next_maximum;
  }

  const float inv = 1.0f / denominator;
  for (int d = 0; d < head_dim; ++d)
    output[q_head * head_dim + d] = accumulator[d] * inv;
}

std::vector<float> cpu_materialized_reference(const std::vector<float>& query,
                                              const std::vector<float>& keys,
                                              const std::vector<float>& values,
                                              int tokens,
                                              int query_heads,
                                              int kv_heads,
                                              int head_dim) {
  std::vector<float> output(query_heads * head_dim, 0.0f);
  const int group_size = query_heads / kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  for (int q_head = 0; q_head < query_heads; ++q_head) {
    const int kv_head = q_head / group_size;
    std::vector<double> scores(tokens);
    for (int t = 0; t < tokens; ++t) {
      double score = 0.0;
      for (int d = 0; d < head_dim; ++d)
        score += static_cast<double>(query[q_head * head_dim + d]) *
                 keys[(t * kv_heads + kv_head) * head_dim + d];
      scores[t] = score * scale;
    }

    const double maximum = *std::max_element(scores.begin(), scores.end());
    double denominator = 0.0;
    for (double score : scores) denominator += std::exp(score - maximum);

    for (int t = 0; t < tokens; ++t) {
      const double probability = std::exp(scores[t] - maximum) / denominator;
      for (int d = 0; d < head_dim; ++d)
        output[q_head * head_dim + d] +=
            static_cast<float>(probability * values[(t * kv_heads + kv_head) * head_dim + d]);
    }
  }
  return output;
}

bool validate_close(const std::vector<float>& expected, const std::vector<float>& actual,
                    float tolerance) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float scale = std::max({1.0f, std::abs(expected[i]), std::abs(actual[i])});
    if (std::abs(expected[i] - actual[i]) > tolerance * scale) {
      std::cerr << "mismatch at index " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int tokens = 3;
  constexpr int query_heads = 2;
  constexpr int kv_heads = 1;
  constexpr int head_dim = 2;
  constexpr int token_elements = kv_heads * head_dim;

  const std::vector<float> all_k{
      1.0f, 0.0f,
      0.5f, -1.0f,
      1.5f, 0.25f};
  const std::vector<float> all_v{
      2.0f, 1.0f,
      -1.0f, 3.0f,
      4.0f, -2.0f};
  const std::vector<float> query{1.0f, -0.5f, 0.25f, 1.5f};

  const std::vector<float> expected =
      cpu_materialized_reference(query, all_k, all_v, tokens, query_heads, kv_heads, head_dim);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_cache_k = nullptr, *d_cache_v = nullptr, *d_query = nullptr, *d_output = nullptr;
  float *d_token_k = nullptr, *d_token_v = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cache_k, tokens * token_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache_v, tokens * token_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_query, query.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, actual.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_k, token_elements * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_v, token_elements * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_query, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
  for (int t = 0; t < tokens; ++t) {
    CUDA_CHECK(cudaMemcpy(d_token_k, all_k.data() + t * token_elements,
                          token_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_token_v, all_v.data() + t * token_elements,
                          token_elements * sizeof(float), cudaMemcpyHostToDevice));
    append_kv_kernel<<<(token_elements + kThreads - 1) / kThreads, kThreads>>>(
        d_cache_k, d_cache_v, d_token_k, d_token_v, tokens, kv_heads, head_dim, t);
    CUDA_CHECK(cudaGetLastError());
  }

  cached_single_token_attention_kernel<<<(query_heads + kThreads - 1) / kThreads, kThreads>>>(
      d_query, d_cache_k, d_cache_v, d_output, tokens, query_heads, kv_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_cache_k));
  CUDA_CHECK(cudaFree(d_cache_v));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaFree(d_token_k));
  CUDA_CHECK(cudaFree(d_token_v));

  if (!validate_close(expected, actual, 2e-4f)) return 1;

  std::cout << "p023 CUDA canonical solution passed cached single-token attention validation\n";
  return 0;
}
