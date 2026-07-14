#include "cuda_check.hpp"

#include <iostream>
#include <vector>

namespace {

__global__ void ring_append_todo_kernel(int* slots, int capacity,
                                        int start_position, int append_count) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= append_count) return;

  // TODO(p026): implement fixed-capacity wraparound writes and chronological reads.
  const int logical_position = start_position + token;
  const int wrong_slot = token;  // intentionally ignores modulo wrap.
  if (wrong_slot < capacity) slots[wrong_slot] = logical_position;
}

}  // namespace

int main() {
  constexpr int capacity = 3;
  constexpr int append_count = 8;

  int* d_slots = nullptr;
  CUDA_CHECK(cudaMalloc(&d_slots, capacity * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_slots, 0xff, capacity * sizeof(int)));
  ring_append_todo_kernel<<<1, 64>>>(d_slots, capacity, 10, append_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_slots));

  std::cerr << "p026 starter intentionally fails: TODO ring overwrite and chronological ordering\n";
  return 1;
}
