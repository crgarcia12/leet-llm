#include "cuda_check.hpp"

#include <iostream>
#include <vector>

__global__ void append_kv_kernel(const float* residual, const float* wk, const float* wv,
                                 float* cache_k, float* cache_v, int context_before, int dim) {
  const int out = blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= dim) return;
  cache_k[context_before * dim + out] = residual[out];
  cache_v[context_before * dim + out] = residual[out];
  (void)wk;
  (void)wv;
  // TODO: project residual through Wk/Wv before append.
}

int main() {
  constexpr int dim = 4;
  std::vector<float> residual{0.5f, 1.0f, -0.25f, 0.75f};
  std::vector<float> cache((2 + 1) * dim, 0.0f);
  float *d_residual = nullptr, *d_cache = nullptr;
  CUDA_CHECK(cudaMalloc(&d_residual, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cache, cache.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_residual, residual.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  append_kv_kernel<<<1, 128>>>(d_residual, nullptr, nullptr, d_cache, d_cache, 2, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_cache));
  std::cout << "p040 starter builds. TODO: cached attention and sampling update are incomplete.\n";
  return 0;
}
