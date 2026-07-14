#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void multi_head_attention_todo_kernel(const float* query,
                                                 const float* keys,
                                                 const float* values,
                                                 float* output,
                                                 int query_length,
                                                 int key_length,
                                                 int heads,
                                                 int head_dim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_rows = query_length * heads;
  if (row >= total_rows) return;

  const int q = row / heads;
  const int h = row % heads;
  // TODO(p017): keep heads independent. This placeholder leaks head-0 K/V to all heads.
  for (int d = 0; d < head_dim; ++d)
    output[(q * heads + h) * head_dim + d] = values[d] + query[(q * heads + h) * head_dim + d];
}

}  // namespace

int main() {
  const int query_length = 1;
  const int key_length = 1;
  const int heads = 2;
  const int head_dim = 1;
  const std::vector<float> query{1.0f, 0.0f};
  const std::vector<float> keys{1.0f, 1.0f};
  const std::vector<float> values{2.0f, 8.0f};
  std::vector<float> output(2, 0.0f);

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, query.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, keys.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  multi_head_attention_todo_kernel<<<1, 64>>>(d_q, d_k, d_v, d_o,
                                              query_length, key_length, heads, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_o, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  const bool passed = std::abs(output[0] - 2.0f) <= 1e-5f && std::abs(output[1] - 8.0f) <= 1e-5f;
  if (passed) {
    std::cerr << "p017 starter unexpectedly passed; TODO head isolation is missing\n";
    return 1;
  }
  std::cerr << "p017 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
