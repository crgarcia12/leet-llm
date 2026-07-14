#include "cuda_check.hpp"

#include <iostream>
#include <vector>

__global__ void fused_qkv_starter_kernel(const float* input, const float* wq,
                                         float* query, int sequence, int model_dim, int query_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= sequence * query_dim) return;
  const int token = idx / query_dim;
  const int out = idx % query_dim;
  float sum = 0.0f;
  for (int i = 0; i < model_dim; ++i) sum += input[token * model_dim + i] * wq[out * model_dim + i];
  query[idx] = sum;
  // TODO: fuse RMSNorm first and include key/value projections in one fused launch.
}

int main() {
  std::cout << "p043 starter builds. TODO: fused RMSNorm + Q/K/V path is incomplete.\n";
  return 0;
}
