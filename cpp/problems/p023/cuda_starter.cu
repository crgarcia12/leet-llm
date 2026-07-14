#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void cached_attention_todo_kernel(const float* cache_v,
                                             float* output,
                                             int tokens,
                                             int head_dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < head_dim) {
    // TODO(p023): run stable attention over all cached keys/values.
    output[index] = cache_v[(tokens - 1) * head_dim + index];
  }
}

}  // namespace

int main() {
  const int tokens = 2;
  const int head_dim = 1;
  const std::vector<float> cache_v{2.0f, 4.0f};
  std::vector<float> output(1, 0.0f);

  float *d_cache_v = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cache_v, cache_v.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_cache_v, cache_v.data(), cache_v.size() * sizeof(float), cudaMemcpyHostToDevice));

  cached_attention_todo_kernel<<<1, 32>>>(d_cache_v, d_output, tokens, head_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_cache_v));
  CUDA_CHECK(cudaFree(d_output));

  const float expected = 2.537883f;
  if (std::abs(output[0] - expected) <= 1e-4f) {
    std::cerr << "p023 starter unexpectedly passed; TODO cached attention recurrence is missing\n";
    return 1;
  }
  std::cerr << "p023 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
