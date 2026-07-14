#include "cuda_check.hpp"

#include <array>
#include <iostream>

namespace {

__global__ void paged_gather_kernel(const float* physical_pages, const int* page_table,
                                    float* gathered, int page_size, int token_count) {
  const int logical_slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (logical_slot >= token_count) return;
  const int page_ordinal = logical_slot / page_size;
  const int slot_in_page = logical_slot % page_size;
  const int physical_page = page_table[page_ordinal];
  gathered[logical_slot] = physical_pages[physical_page * page_size + slot_in_page];
}

}  // namespace

int main() {
  constexpr int page_size = 2;
  constexpr int token_count = 4;
  constexpr int total_pages = 3;

  const std::array<float, total_pages * page_size> physical_pages{
      10.0f, 11.0f,
      99.0f, 98.0f,
      12.0f, 13.0f,
  };
  const std::array<int, 2> page_table{0, 2};

  float* d_pages = nullptr;
  int* d_table = nullptr;
  float* d_gathered = nullptr;
  CUDA_CHECK(cudaMalloc(&d_pages, physical_pages.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_table, page_table.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_gathered, token_count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_pages, physical_pages.data(), physical_pages.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_table, page_table.data(), page_table.size() * sizeof(int), cudaMemcpyHostToDevice));

  paged_gather_kernel<<<1, 64>>>(d_pages, d_table, d_gathered, page_size, token_count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::array<float, token_count> gathered{};
  CUDA_CHECK(cudaMemcpy(gathered.data(), d_gathered, token_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_pages));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_gathered));

  const std::array<float, token_count> expected{10.0f, 11.0f, 12.0f, 13.0f};
  if (gathered != expected) return 1;

  std::cout << "p027 CUDA paged KV gather passed\n";
  return 0;
}
