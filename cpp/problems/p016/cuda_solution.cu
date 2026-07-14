#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void causal_attention_one_head_kernel(const float* query,
                                                 const float* keys,
                                                 const float* values,
                                                 float* output,
                                                 int query_length,
                                                 int key_length,
                                                 int head_dim,
                                                 int query_offset,
                                                 int key_offset) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= query_length) return;

  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int query_position = query_offset + q;
  float max_score = -CUDART_INF_F;
  for (int k = 0; k < key_length; ++k) {
    const int key_position = key_offset + k;
    if (key_position > query_position) continue;

    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d)
      score += query[q * head_dim + d] * keys[k * head_dim + d];
    score *= scale;
    max_score = fmaxf(max_score, score);
  }

  float denominator = 0.0f;
  for (int d = 0; d < head_dim; ++d) output[q * head_dim + d] = 0.0f;

  for (int k = 0; k < key_length; ++k) {
    const int key_position = key_offset + k;
    if (key_position > query_position) continue;

    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d)
      score += query[q * head_dim + d] * keys[k * head_dim + d];
    const float weight = expf(score * scale - max_score);
    denominator += weight;
    for (int d = 0; d < head_dim; ++d)
      output[q * head_dim + d] += weight * values[k * head_dim + d];
  }

  const float inv = 1.0f / denominator;
  for (int d = 0; d < head_dim; ++d) output[q * head_dim + d] *= inv;
}

std::vector<float> cpu_reference(const std::vector<float>& query,
                                 const std::vector<float>& keys,
                                 const std::vector<float>& values,
                                 int query_length,
                                 int key_length,
                                 int head_dim,
                                 int query_offset,
                                 int key_offset) {
  std::vector<float> output(query_length * head_dim, 0.0f);
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  for (int q = 0; q < query_length; ++q) {
    const int query_position = query_offset + q;
    std::vector<double> masked_scores;
    std::vector<int> visible_keys;
    for (int k = 0; k < key_length; ++k) {
      const int key_position = key_offset + k;
      if (key_position > query_position) continue;

      double score = 0.0;
      for (int d = 0; d < head_dim; ++d)
        score += static_cast<double>(query[q * head_dim + d]) * keys[k * head_dim + d];
      masked_scores.push_back(score * scale);
      visible_keys.push_back(k);
    }

    const double maximum = *std::max_element(masked_scores.begin(), masked_scores.end());
    double denominator = 0.0;
    for (double score : masked_scores) denominator += std::exp(score - maximum);

    for (std::size_t i = 0; i < visible_keys.size(); ++i) {
      const double probability = std::exp(masked_scores[i] - maximum) / denominator;
      const int k = visible_keys[i];
      for (int d = 0; d < head_dim; ++d)
        output[q * head_dim + d] += static_cast<float>(probability * values[k * head_dim + d]);
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
  constexpr int query_length = 3;
  constexpr int key_length = 4;
  constexpr int head_dim = 2;

  const std::vector<float> query{
      1.0f, 0.5f,
      -0.5f, 2.0f,
      1.5f, -1.0f};
  const std::vector<float> keys{
      1.0f, 0.0f,
      0.0f, 1.0f,
      1.0f, 1.0f,
      -1.0f, 0.5f};
  const std::vector<float> values{
      2.0f, 0.0f,
      0.0f, 3.0f,
      4.0f, -1.0f,
      -2.0f, 1.0f};

  const std::vector<float> expected = cpu_reference(query, keys, values, query_length,
                                                    key_length, head_dim, 2, 0);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, query.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, keys.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, actual.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  causal_attention_one_head_kernel<<<(query_length + kThreads - 1) / kThreads, kThreads>>>(
      d_q, d_k, d_v, d_o, query_length, key_length, head_dim, 2, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_o, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  if (!validate_close(expected, actual, 8e-5f)) return 1;

  std::cout << "p016 CUDA canonical solution passed one-head causal attention validation\n";
  return 0;
}
