#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void causal_attention_todo_kernel(const float* query, const float* keys,
                                             const float* values, float* output,
                                             int query_length, int key_length,
                                             int head_dim) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= query_length) return;

  // TODO(p016): apply causal mask and stable softmax before mixing values.
  for (int d = 0; d < head_dim; ++d) {
    float sum = 0.0f;
    for (int k = 0; k < key_length; ++k) sum += values[k * head_dim + d];
    output[q * head_dim + d] = sum / static_cast<float>(key_length);
  }
}

}  // namespace

int main() {
  constexpr int query_length = 1;
  constexpr int key_length = 2;
  constexpr int head_dim = 1;
  const std::vector<float> query{1.0f};
  const std::vector<float> keys{1.0f, 2.0f};
  const std::vector<float> values{3.0f, 9.0f};
  std::vector<float> output(1, 0.0f);

  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q, query.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, keys.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_o, output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_q, query.data(), query.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  causal_attention_todo_kernel<<<1, 64>>>(d_q, d_k, d_v, d_o, query_length, key_length, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_o, output.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));
  CUDA_CHECK(cudaFree(d_o));

  const float expected = 3.0f;  // Query at first position should only see first value.
  if (std::abs(output[0] - expected) <= 1e-4f) {
    std::cerr << "p016 starter unexpectedly passed; TODO causal masking is missing\n";
    return 1;
  }
  std::cerr << "p016 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
