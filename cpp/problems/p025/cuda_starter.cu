#include "cuda_check.hpp"

#include <iostream>
#include <vector>

namespace {

__global__ void shared_kv_heads_todo_kernel(const float* query, const float* cached_values,
                                            float* output, int query_heads, int kv_heads,
                                            int head_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = query_heads * head_dim;
  if (index >= count) return;

  // TODO(p025): replace this modulo mapping with contiguous-group division mapping.
  const int query_head = index / head_dim;
  const int dim = index % head_dim;
  const int wrong_kv_head = query_head % kv_heads;
  output[index] = query[index] + cached_values[wrong_kv_head * head_dim + dim];
}

}  // namespace

int main() {
  constexpr int query_heads = 4, kv_heads = 2, head_dim = 3;
  const int count = query_heads * head_dim;

  std::vector<float> query(count, 1.0f);
  std::vector<float> cached_values{100.0f, 200.0f, 300.0f, 1000.0f, 2000.0f, 3000.0f};
  std::vector<float> output(count, 0.0f);

  float *d_query = nullptr, *d_values = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_query, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_values, kv_heads * head_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_query, query.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, cached_values.data(), kv_heads * head_dim * sizeof(float), cudaMemcpyHostToDevice));

  shared_kv_heads_todo_kernel<<<1, 128>>>(d_query, d_values, d_output, query_heads, kv_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, count * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_query));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_output));

  std::cerr << "p025 starter intentionally fails: implement contiguous kvHead = qHead / groupSize mapping\n";
  return 1;
}
