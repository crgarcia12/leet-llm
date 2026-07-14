#include "cuda_check.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>

struct Lifetime {
  int first;
  int last;
  int bytes;
  int alignment;
};

__global__ void overlap_kernel(const int* first, const int* last, int count, int* overlap) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = count * count;
  if (idx >= total) return;
  const int i = idx / count;
  const int j = idx % count;
  overlap[idx] = (first[i] <= last[j] && first[j] <= last[i]) ? 1 : 0;
}

int align_up(int value, int alignment) { return (value + alignment - 1) & ~(alignment - 1); }

int main() {
  const std::array<Lifetime, 4> life{{{0, 2, 24, 8}, {1, 1, 8, 16}, {2, 4, 16, 8}, {3, 3, 20, 4}}};
  std::vector<int> first{0, 1, 2, 3}, last{2, 1, 4, 3};
  int *d_first = nullptr, *d_last = nullptr, *d_overlap = nullptr;
  CUDA_CHECK(cudaMalloc(&d_first, first.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_last, last.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_overlap, first.size() * last.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_first, first.data(), first.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_last, last.data(), last.size() * sizeof(int), cudaMemcpyHostToDevice));
  overlap_kernel<<<1, 128>>>(d_first, d_last, static_cast<int>(life.size()), d_overlap);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int> overlap(life.size() * life.size());
  CUDA_CHECK(cudaMemcpy(overlap.data(), d_overlap, overlap.size() * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_first));
  CUDA_CHECK(cudaFree(d_last));
  CUDA_CHECK(cudaFree(d_overlap));

  std::vector<int> offsets(life.size(), 0);
  int arena = 0;
  for (int i = 0; i < static_cast<int>(life.size()); ++i) {
    int candidate = 0;
    for (;;) {
      candidate = align_up(candidate, life[i].alignment);
      bool blocked = false;
      for (int j = 0; j < i; ++j) {
        if (!overlap[i * life.size() + j]) continue;
        const int other_end = offsets[j] + life[j].bytes;
        if (candidate < other_end && offsets[j] < candidate + life[i].bytes) {
          candidate = other_end;
          blocked = true;
          break;
        }
      }
      if (!blocked) break;
    }
    offsets[i] = candidate;
    arena = std::max(arena, candidate + life[i].bytes);
  }

  const std::vector<int> expected{0, 32, 24, 0};
  if (offsets != expected || arena != 40) return 1;
  std::cout << "p041 first-fit arena reuse validated with CUDA overlap analysis\n";
  return 0;
}
