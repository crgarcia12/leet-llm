#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void grouped_query_attention_todo_kernel(const float* values,
                                                    float* output,
                                                    int query_heads,
                                                    int kv_heads,
                                                    int head_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = query_heads * head_dim;
  if (index >= total) return;

  const int q_head = index / head_dim;
  const int feature = index % head_dim;
  // TODO(p018): use contiguous-group mapping kvHead = queryHead / groupSize.
  const int wrong_kv_head = q_head % kv_heads;
  output[index] = values[wrong_kv_head * head_dim + feature];
}

}  // namespace

int main() {
  const int query_heads = 4;
  const int kv_heads = 2;
  const int head_dim = 1;
  const std::vector<float> values{10.0f, 20.0f};
  std::vector<float> output(query_heads * head_dim, 0.0f);

  float *d_values = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  grouped_query_attention_todo_kernel<<<1, 64>>>(d_values, d_output, query_heads, kv_heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_output));

  const std::vector<float> expected{10.0f, 10.0f, 20.0f, 20.0f};
  bool passed = true;
  for (int i = 0; i < query_heads; ++i)
    passed = passed && std::abs(output[i] - expected[i]) <= 1e-5f;
  if (passed) {
    std::cerr << "p018 starter unexpectedly passed; TODO contiguous GQA mapping is missing\n";
    return 1;
  }
  std::cerr << "p018 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
