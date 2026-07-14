#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;
constexpr int kTile = 4;

__global__ void tiled_online_attention_kernel(const float* query,
                                              const float* keys,
                                              const float* values,
                                              float* output,
                                              int query_length,
                                              int key_length,
                                              int query_heads,
                                              int kv_heads,
                                              int head_dim,
                                              int query_offset,
                                              int key_offset) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = query_length * query_heads;
  if (row >= total_rows) return;

  const int q = row / query_heads;
  const int q_head = row % query_heads;
  const int group_size = query_heads / kv_heads;
  const int kv_head = q_head / group_size;
  const int query_position = query_offset + q;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  constexpr int kMaxHeadDim = 8;
  float accumulator[kMaxHeadDim] = {0.0f};
  float running_max = -CUDART_INF_F;
  float running_denominator = 0.0f;

  for (int tile_start = 0; tile_start < key_length; tile_start += kTile) {
    const int tile_end = min(tile_start + kTile, key_length);

    float tile_max = -CUDART_INF_F;
    float tile_scores[kTile];
    bool valid[kTile];

    for (int i = 0; i < kTile; ++i) {
      tile_scores[i] = -CUDART_INF_F;
      valid[i] = false;
    }

    for (int k = tile_start; k < tile_end; ++k) {
      const int local = k - tile_start;
      if (key_offset + k > query_position) continue;
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d)
        score += query[(q * query_heads + q_head) * head_dim + d] *
                 keys[(k * kv_heads + kv_head) * head_dim + d];
      score *= scale;
      tile_scores[local] = score;
      valid[local] = true;
      tile_max = fmaxf(tile_max, score);
    }

    if (tile_max == -CUDART_INF_F) continue;

    const float next_max = fmaxf(running_max, tile_max);
    const float alpha = (running_max == -CUDART_INF_F) ? 0.0f : expf(running_max - next_max);

    float tile_denominator = 0.0f;
    float tile_weighted[kMaxHeadDim] = {0.0f};
    for (int local = 0; local < (tile_end - tile_start); ++local) {
      if (!valid[local]) continue;
      const int k = tile_start + local;
      const float weight = expf(tile_scores[local] - next_max);
      tile_denominator += weight;
      for (int d = 0; d < head_dim; ++d)
        tile_weighted[d] += weight * values[(k * kv_heads + kv_head) * head_dim + d];
    }

    running_denominator = running_denominator * alpha + tile_denominator;
    for (int d = 0; d < head_dim; ++d)
      accumulator[d] = accumulator[d] * alpha + tile_weighted[d];
    running_max = next_max;
  }

  const float inv = 1.0f / running_denominator;
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
                                              int key_offset) {
  std::vector<float> output(query_length * query_heads * head_dim, 0.0f);
  const int group_size = query_heads / kv_heads;
  const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

  for (int q = 0; q < query_length; ++q) {
    const int query_position = query_offset + q;
    for (int q_head = 0; q_head < query_heads; ++q_head) {
      const int kv_head = q_head / group_size;
      std::vector<double> scores;
      std::vector<int> visible_keys;

      for (int k = 0; k < key_length; ++k) {
        if (key_offset + k > query_position) continue;
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
  constexpr int query_length = 2;
  constexpr int key_length = 7;
  constexpr int query_heads = 2;
  constexpr int kv_heads = 1;
  constexpr int head_dim = 3;

  const std::vector<float> query{
      0.5f, 1.0f, -1.0f, 1.5f, -0.5f, 0.25f,
      -1.0f, 2.0f, 0.5f, 0.25f, -0.75f, 1.25f};
  const std::vector<float> keys{
      1.0f, 0.0f, 0.5f,
      -1.0f, 2.0f, 0.25f,
      0.5f, -0.5f, 1.5f,
      2.0f, 1.0f, -1.0f,
      -0.25f, 0.75f, 1.0f,
      1.25f, -1.5f, 0.5f,
      0.0f, 1.0f, -0.5f};
  const std::vector<float> values{
      2.0f, 0.0f, -1.0f,
      1.0f, 3.0f, 0.5f,
      -2.0f, 1.5f, 2.0f,
      0.25f, -0.5f, 4.0f,
      1.5f, 2.5f, -3.0f,
      -1.0f, 0.5f, 1.0f,
      3.0f, -2.0f, 0.75f};

  const std::vector<float> expected = cpu_materialized_reference(
      query, keys, values, query_length, key_length, query_heads, kv_heads,
      head_dim, 4, 0);
  std::vector<float> actual(expected.size(), 0.0f);

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, query.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, keys.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, actual.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  tiled_online_attention_kernel<<<(query_length * query_heads + kThreads - 1) / kThreads, kThreads>>>(
      d_q, d_k, d_v, d_o, query_length, key_length, query_heads, kv_heads,
      head_dim, 4, 0);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_o, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  if (!validate_close(expected, actual, 2.5e-4f)) return 1;

  std::cout << "p020 CUDA canonical solution passed tiled online-attention validation\n";
  return 0;
}
