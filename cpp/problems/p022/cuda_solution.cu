#include "cuda_check.hpp"

#include <algorithm>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void append_kv_kernel(float* cache_k, float* cache_v,
                                 const float* token_k, const float* token_v,
                                 int layers, int capacity, int kv_heads,
                                 int head_dim, int layer, int slot) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int token_elements = kv_heads * head_dim;
  if (index >= token_elements) return;

  const int cache_index = (((layer * capacity + slot) * kv_heads) * head_dim) + index;
  cache_k[cache_index] = token_k[index];
  cache_v[cache_index] = token_v[index];
}

__global__ void read_head_vector_kernel(const float* cache, float* out,
                                        int capacity, int kv_heads, int head_dim,
                                        int layer, int slot, int head) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= head_dim) return;
  const int cache_index = (((layer * capacity + slot) * kv_heads + head) * head_dim) + d;
  out[d] = cache[cache_index];
}

std::vector<float> cpu_cache_reference(const std::vector<float>& appended,
                                       int layers, int capacity,
                                       int kv_heads, int head_dim,
                                       int layer, int slot) {
  std::vector<float> cache(layers * capacity * kv_heads * head_dim, 0.0f);
  const int base = (((layer * capacity + slot) * kv_heads) * head_dim);
  for (int i = 0; i < kv_heads * head_dim; ++i) cache[base + i] = appended[i];
  return cache;
}

bool validate_close(const std::vector<float>& expected, const std::vector<float>& actual,
                    float tolerance) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::abs(expected[i] - actual[i]) > tolerance) {
      std::cerr << "mismatch at index " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int layers = 2;
  constexpr int capacity = 5;
  constexpr int kv_heads = 2;
  constexpr int head_dim = 3;
  constexpr int token_elements = kv_heads * head_dim;

  const std::vector<float> token_k{1, 2, 3, 4, 5, 6};
  const std::vector<float> token_v{-1, -2, -3, 7, 8, 9};

  std::vector<float> expected_k = cpu_cache_reference(token_k, layers, capacity, kv_heads, head_dim, 1, 2);
  std::vector<float> expected_v = cpu_cache_reference(token_v, layers, capacity, kv_heads, head_dim, 1, 2);
  std::vector<float> actual_k(expected_k.size(), 0.0f);
  std::vector<float> actual_v(expected_v.size(), 0.0f);
  std::vector<float> readback(head_dim, 0.0f);

  float *d_cache_k = nullptr, *d_cache_v = nullptr, *d_token_k = nullptr, *d_token_v = nullptr;
  float* d_read = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cache_k, actual_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache_v, actual_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_k, token_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_v, token_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_read, readback.size() * sizeof(float)));

  CUDA_CHECK(cudaMemset(d_cache_k, 0, actual_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_cache_v, 0, actual_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_token_k, token_k.data(), token_k.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_token_v, token_v.data(), token_v.size() * sizeof(float), cudaMemcpyHostToDevice));

  append_kv_kernel<<<(token_elements + kThreads - 1) / kThreads, kThreads>>>(
      d_cache_k, d_cache_v, d_token_k, d_token_v, layers, capacity, kv_heads,
      head_dim, 1, 2);
  CUDA_CHECK(cudaGetLastError());

  read_head_vector_kernel<<<(head_dim + kThreads - 1) / kThreads, kThreads>>>(
      d_cache_v, d_read, capacity, kv_heads, head_dim, 1, 2, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual_k.data(), d_cache_k, actual_k.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual_v.data(), d_cache_v, actual_v.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(readback.data(), d_read, readback.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_cache_k));
  CUDA_CHECK(cudaFree(d_cache_v));
  CUDA_CHECK(cudaFree(d_token_k));
  CUDA_CHECK(cudaFree(d_token_v));
  CUDA_CHECK(cudaFree(d_read));

  const std::vector<float> expected_head{7, 8, 9};
  if (!validate_close(expected_k, actual_k, 1e-6f) ||
      !validate_close(expected_v, actual_v, 1e-6f) ||
      !validate_close(expected_head, readback, 1e-6f))
    return 1;

  std::cout << "p022 CUDA canonical solution passed preallocated KV append/read validation\n";
  return 0;
}
