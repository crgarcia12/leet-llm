#include "cuda_check.hpp"

#include <array>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 64;

__global__ void ring_append_kernel(int* slots, int capacity,
                                   int start_position, int append_count) {
  const int token = blockIdx.x * blockDim.x + threadIdx.x;
  if (token >= append_count) return;
  const int logical_position = start_position + token;
  const int slot = token % capacity;
  slots[slot] = logical_position;
}

__global__ void ring_chronological_kernel(const int* slots, int* chronological,
                                          int capacity, int next_slot) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= capacity) return;
  const int physical = (next_slot + idx) % capacity;
  chronological[idx] = slots[physical];
}

}  // namespace

int main() {
  constexpr int capacity = 3;
  constexpr int start_position = 10;
  constexpr int append_count = 8;

  int* d_slots = nullptr;
  int* d_chronological = nullptr;
  CUDA_CHECK(cudaMalloc(&d_slots, capacity * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_chronological, capacity * sizeof(int)));
  CUDA_CHECK(cudaMemset(d_slots, 0xff, capacity * sizeof(int)));

  ring_append_kernel<<<1, kThreads>>>(d_slots, capacity, start_position, append_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  const int next_slot = append_count % capacity;
  ring_chronological_kernel<<<1, kThreads>>>(d_slots, d_chronological, capacity, next_slot);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::array<int, capacity> slots{};
  std::array<int, capacity> chronological{};
  CUDA_CHECK(cudaMemcpy(slots.data(), d_slots, capacity * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(chronological.data(), d_chronological, capacity * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_slots));
  CUDA_CHECK(cudaFree(d_chronological));

  const std::array<int, capacity> expected_slots{16, 17, 15};
  const std::array<int, capacity> expected_chronological{15, 16, 17};
  if (slots != expected_slots || chronological != expected_chronological) return 1;

  std::cout << "p026 CUDA ring-buffer chronology passed\n";
  return 0;
}
