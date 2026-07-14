#include "cuda_check.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void shared_kv_heads_kernel(const float* query, const float* cached_values,
                                       float* output, int query_heads, int kv_heads,
                                       int head_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = query_heads * head_dim;
  if (index >= count) return;
  const int query_head = index / head_dim;
  const int dim = index % head_dim;
  const int group_size = query_heads / kv_heads;
  const int kv_head = query_head / group_size;
  output[index] = query[index] + cached_values[kv_head * head_dim + dim];
}

bool near(float a, float b, float eps = 1e-6f) { return std::fabs(a - b) <= eps; }

}  // namespace

int main() {
  constexpr int query_heads = 4;
  constexpr int kv_heads = 2;
  constexpr int head_dim = 3;
  const int count = query_heads * head_dim;

  const std::vector<float> query{
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
      10.0f, 11.0f, 12.0f,
  };
  const std::vector<float> cached_values{
      100.0f, 200.0f, 300.0f,
      1000.0f, 2000.0f, 3000.0f,
  };

  std::vector<float> expected(count, 0.0f);
  for (int qh = 0; qh < query_heads; ++qh) {
    const int kvh = qh / (query_heads / kv_heads);
    for (int d = 0; d < head_dim; ++d)
      expected[qh * head_dim + d] = query[qh * head_dim + d] + cached_values[kvh * head_dim + d];
  }

  float *d_query = nullptr, *d_values = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_values, kv_heads * head_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_query, query.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, cached_values.data(), kv_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));

  shared_kv_heads_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
      d_query, d_values, d_output, query_heads, kv_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(count, 0.0f);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_output));

  const std::array<int, 4> mapping{0, 0, 1, 1};
  for (int i = 0; i < 4; ++i)
    if (mapping[i] != i / 2) return 1;

  for (int i = 0; i < count; ++i)
    if (!near(output[i], expected[i])) return 1;

  std::cout << "p025 CUDA shared KV-head mapping passed\n";
  return 0;
}
