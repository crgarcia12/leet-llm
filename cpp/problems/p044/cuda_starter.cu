#include "cuda_check.hpp"

#include <iostream>

__global__ void workload_kernel(float* buffer, int count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) buffer[idx] += 1.0f;
}

int main() {
  // TODO: add warmup/measured loops, per-stage sample collection, and percentile summaries.
  std::cout << "p044 starter builds. TODO: profiling loop and statistics are incomplete.\n";
  return 0;
}
