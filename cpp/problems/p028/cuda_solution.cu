#include "cuda_check.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__global__ void quantize_kv_kernel(const float* k_in, const float* v_in,
                                   std::int8_t* k_q, std::int8_t* v_q,
                                   float* k_scales, float* v_scales,
                                   int token_count, int kv_heads, int head_dim) {
  const int vector_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int vector_count = token_count * kv_heads;
  if (vector_index >= vector_count) return;

  const int base = vector_index * head_dim;
  float max_abs_k = 0.0f;
  float max_abs_v = 0.0f;
  for (int d = 0; d < head_dim; ++d) {
    max_abs_k = fmaxf(max_abs_k, fabsf(k_in[base + d]));
    max_abs_v = fmaxf(max_abs_v, fabsf(v_in[base + d]));
  }
  const float k_scale = max_abs_k == 0.0f ? 1.0f : max_abs_k / 127.0f;
  const float v_scale = max_abs_v == 0.0f ? 1.0f : max_abs_v / 127.0f;
  k_scales[vector_index] = k_scale;
  v_scales[vector_index] = v_scale;

  for (int d = 0; d < head_dim; ++d) {
    k_q[base + d] = static_cast<std::int8_t>(llrintf(fminf(127.0f, fmaxf(-127.0f, k_in[base + d] / k_scale))));
    v_q[base + d] = static_cast<std::int8_t>(llrintf(fminf(127.0f, fmaxf(-127.0f, v_in[base + d] / v_scale))));
  }
}

__global__ void dequantized_dot_kernel(const std::int8_t* k_q, const float* k_scales,
                                       const float* query, float* scores,
                                       int token_count, int kv_heads, int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = token_count * kv_heads;
  if (idx >= count) return;
  const int base = idx * head_dim;
  float score = 0.0f;
  for (int d = 0; d < head_dim; ++d)
    score += static_cast<float>(k_q[base + d]) * k_scales[idx] * query[d];
  scores[idx] = score;
}

bool near(float a, float b, float eps = 2e-3f) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
  constexpr int token_count = 3;
  constexpr int kv_heads = 2;
  constexpr int head_dim = 4;
  constexpr int vector_count = token_count * kv_heads;
  constexpr int elem_count = vector_count * head_dim;

  std::vector<float> k(elem_count), v(elem_count);
  for (int i = 0; i < elem_count; ++i) {
    k[i] = static_cast<float>((i % 7) - 3) * 0.5f;
    v[i] = static_cast<float>((i % 5) - 2) * 1.25f;
  }
  const std::vector<float> query{0.5f, -1.0f, 1.5f, 0.25f};

  float *d_k = nullptr, *d_v = nullptr, *d_query = nullptr;
  std::int8_t *d_kq = nullptr, *d_vq = nullptr;
  float *d_k_scales = nullptr, *d_v_scales = nullptr, *d_scores = nullptr;

  CUDA_CHECK(cudaMalloc(&d_k, elem_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, elem_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_query, head_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_kq, elem_count * sizeof(std::int8_t)));
  CUDA_CHECK(cudaMalloc(&d_vq, elem_count * sizeof(std::int8_t)));
  CUDA_CHECK(cudaMalloc(&d_k_scales, vector_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v_scales, vector_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_scores, vector_count * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_k, k.data(), elem_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, v.data(), elem_count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_query, query.data(), head_dim * sizeof(float), cudaMemcpyHostToDevice));

  quantize_kv_kernel<<<1, 64>>>(d_k, d_v, d_kq, d_vq, d_k_scales, d_v_scales, token_count, kv_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  dequantized_dot_kernel<<<1, 64>>>(d_kq, d_k_scales, d_query, d_scores, token_count, kv_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> k_scales(vector_count), v_scales(vector_count), scores(vector_count);
  CUDA_CHECK(cudaMemcpy(k_scales.data(), d_k_scales, vector_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(v_scales.data(), d_v_scales, vector_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, vector_count * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_k)); CUDA_CHECK(cudaFree(d_v)); CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_kq)); CUDA_CHECK(cudaFree(d_vq));
  CUDA_CHECK(cudaFree(d_k_scales)); CUDA_CHECK(cudaFree(d_v_scales)); CUDA_CHECK(cudaFree(d_scores));

  for (int i = 0; i < vector_count; ++i)
    if (!(k_scales[i] > 0.0f && v_scales[i] > 0.0f)) return 1;
  if (near(k_scales[0], v_scales[0])) return 1;

  std::vector<float> expected(vector_count, 0.0f);
  for (int vec = 0; vec < vector_count; ++vec) {
    for (int d = 0; d < head_dim; ++d) {
      const int idx = vec * head_dim + d;
      const float reconstructed = std::round(k[idx] / k_scales[vec]);
      const float clamped = std::fmax(-127.0f, std::fmin(127.0f, reconstructed));
      expected[vec] += clamped * k_scales[vec] * query[d];
    }
    if (!near(expected[vec], scores[vec], 5e-2f)) return 1;
  }

  const std::size_t bytes = 2 * token_count * kv_heads * head_dim * sizeof(std::int8_t) +
                            2 * token_count * kv_heads * sizeof(float);
  if (bytes != 96) return 1;

  std::cout << "p028 CUDA quantized KV cache path passed\n";
  return 0;
}
