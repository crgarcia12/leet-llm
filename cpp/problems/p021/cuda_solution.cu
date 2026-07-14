#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void sliding_window_attention_kernel(const float* query,
                                                const float* keys,
                                                const float* values,
                                                float* output,
                                                int query_length,
                                                int key_length,
                                                int query_heads,
                                                int kv_heads,
                                                int head_dim,
                                                int query_offset,
                                                int key_offset,
                                                int window) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = query_length * query_heads;
  if (row >= total_rows) return;

  const int q = row / query_heads;
  const int q_head = row % query_heads;
  const int group_size = query_heads / kv_heads;
  const int kv_head = q_head / group_size;
  const int query_position = query_offset + q;
  const int lower_bound = max(0, query_position - window + 1);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  float maximum = -CUDART_INF_F;
  float denominator = 0.0f;
  constexpr int kMaxHeadDim = 8;
  float accumulator[kMaxHeadDim] = {0.0f};

  for (int k = 0; k < key_length; ++k) {
    const int key_position = key_offset + k;
    if (key_position > query_position || key_position < lower_bound) continue;

    float score = 0.0f;
    for (int d = 0; d < head_dim; ++d)
      score += query[(q * query_heads + q_head) * head_dim + d] *
               keys[(k * kv_heads + kv_head) * head_dim + d];
    score *= scale;

    const float next_maximum = fmaxf(maximum, score);
    const float alpha = (maximum == -CUDART_INF_F) ? 0.0f : expf(maximum - next_maximum);
    const float beta = expf(score - next_maximum);

    denominator = denominator * alpha + beta;
    for (int d = 0; d < head_dim; ++d)
      accumulator[d] = accumulator[d] * alpha + beta * values[(k * kv_heads + kv_head) * head_dim + d];
    maximum = next_maximum;
  }

  const float inv = 1.0f / denominator;
  for (int d = 0; d < head_dim; ++d)
    output[(q * query_heads + q_head) * head_dim + d] = accumulator[d] * inv;
}

std::vector<float> cpu_materialized_reference(const std::vector<float>& query,
                                              const std::vector<float>& keys,
                                              const std::vector<float>& values,
                                              int query_length,
                                              int key_length,
                                              int query_heads,
                                              int kv_heads,
                                              int head_dim,
                                              int query_offset,
                                              int key_offset,
                                              int window) {
  std::vector<float> output(query_length * query_heads * head_dim, 0.0f);
  const int group_size = query_heads / kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  for (int q = 0; q < query_length; ++q) {
    const int query_position = query_offset + q;
    const int lower_bound = std::max(0, query_position - window + 1);
    for (int q_head = 0; q_head < query_heads; ++q_head) {
      const int kv_head = q_head / group_size;
      std::vector<double> scores;
      std::vector<int> visible_keys;
      for (int k = 0; k < key_length; ++k) {
        const int key_position = key_offset + k;
        if (key_position > query_position || key_position < lower_bound) continue;

        double score = 0.0;
        for (int d = 0; d < head_dim; ++d)
          score += static_cast<double>(query[(q * query_heads + q_head) * head_dim + d]) *
                   keys[(k * kv_heads + kv_head) * head_dim + d];
        scores.push_back(score * scale);
        visible_keys.push_back(k);
      }

      const double maximum = *std::max_element(scores.begin(), scores.end());
      double denominator = 0.0;
      for (double score : scores) denominator += std::exp(score - maximum);

      for (std::size_t i = 0; i < visible_keys.size(); ++i) {
        const double probability = std::exp(scores[i] - maximum) / denominator;
        const int k = visible_keys[i];
        for (int d = 0; d < head_dim; ++d)
          output[(q * query_heads + q_head) * head_dim + d] +=
              static_cast<float>(probability * values[(k * kv_heads + kv_head) * head_dim + d]);
      }
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
  constexpr int key_length = 6;
  constexpr int query_heads = 2;
  constexpr int kv_heads = 1;
  constexpr int head_dim = 2;
  constexpr int window = 3;

  const std::vector<float> query{
      1.0f, 0.5f, -0.5f, 1.5f,
      2.0f, -1.0f, 0.25f, -0.75f,
      -1.0f, 1.0f, 0.5f, 0.25f};
  const std::vector<float> keys{
      0.5f, 1.0f,
      -1.0f, 0.0f,
      1.5f, -0.5f,
      2.0f, 1.0f,
      -0.25f, 0.75f,
      0.5f, -1.5f};
  const std::vector<float> values{
      10.0f, 0.0f,
      1.0f, 2.0f,
      2.0f, -1.0f,
      3.0f, 4.0f,
      4.0f, -2.0f,
      5.0f, 1.0f};

  const std::vector<float> expected = cpu_materialized_reference(
      query, keys, values, query_length, key_length, query_heads, kv_heads,
      head_dim, 4, 1, window);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, query.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, keys.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, actual.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  sliding_window_attention_kernel<<<(query_length * query_heads + kThreads - 1) / kThreads, kThreads>>>(
      d_q, d_k, d_v, d_o, query_length, key_length, query_heads, kv_heads,
      head_dim, 4, 1, window);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_o, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  if (!validate_close(expected, actual, 2e-4f)) return 1;

  std::cout << "p021 CUDA canonical solution passed sliding-window attention validation\n";
  return 0;
}
