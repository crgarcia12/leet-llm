#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void append_kv_todo_kernel(float* cache_k, float* cache_v,
                                      const float* token_k, const float* token_v,
                                      int capacity, int kv_heads, int head_dim,
                                      int slot) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = kv_heads * head_dim;
  if (index >= count) return;

  // TODO(p022): write both K and V using [layer,slot,head,feature] offset.
  const int wrong_index = slot * count + index;
  cache_k[wrong_index] = token_k[index];
  cache_v[wrong_index] = 0.0f * token_v[index];
}

}  // namespace

int main() {
  const int capacity = 2;
  const int kv_heads = 1;
  const int head_dim = 2;
  std::vector<float> cache_k(capacity * kv_heads * head_dim, 0.0f);
  std::vector<float> cache_v(capacity * kv_heads * head_dim, 0.0f);
  const std::vector<float> token_k{1.0f, 2.0f};
  const std::vector<float> token_v{3.0f, 4.0f};

  float *d_cache_k = nullptr, *d_cache_v = nullptr, *d_token_k = nullptr, *d_token_v = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cache_k, cache_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache_v, cache_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_k, token_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_v, token_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_cache_k, 0, cache_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_cache_v, 0, cache_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_token_k, token_k.data(), token_k.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_token_v, token_v.data(), token_v.size() * sizeof(float), cudaMemcpyHostToDevice));

  append_kv_todo_kernel<<<1, 64>>>(d_cache_k, d_cache_v, d_token_k, d_token_v,
                                   capacity, kv_heads, head_dim, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(cache_v.data(), d_cache_v, cache_v.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_cache_k));
  CUDA_CHECK(cudaFree(d_cache_v));
  CUDA_CHECK(cudaFree(d_token_k));
  CUDA_CHECK(cudaFree(d_token_v));

  const bool passed = std::abs(cache_v[2] - 3.0f) <= 1e-5f && std::abs(cache_v[3] - 4.0f) <= 1e-5f;
  if (passed) {
    std::cerr << "p022 starter unexpectedly passed; TODO append path is incomplete\n";
    return 1;
  }
  std::cerr << "p022 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
