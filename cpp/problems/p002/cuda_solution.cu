#include "cuda_check.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

__global__ void gather_strided_kernel(const float* storage, const int* indices, float* output,
                                      int count, int rank,
                                      int d0, int d1, int d2,
                                      int s0, int s1, int s2) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;

  const int i0 = indices[tid * 3 + 0];
  const int i1 = indices[tid * 3 + 1];
  const int i2 = indices[tid * 3 + 2];

  bool in_bounds = true;
  if (rank > 0) in_bounds = in_bounds && (i0 >= 0 && i0 < d0);
  if (rank > 1) in_bounds = in_bounds && (i1 >= 0 && i1 < d1);
  if (rank > 2) in_bounds = in_bounds && (i2 >= 0 && i2 < d2);
  if (!in_bounds) {
    output[tid] = NAN;
    return;
  }

  const int offset = i0 * s0 + i1 * s1 + i2 * s2;
  output[tid] = storage[offset];
}

std::vector<float> cpu_reference(const std::vector<float>& storage,
                                 const std::vector<std::array<int, 3>>& indices,
                                 int rank, int s0, int s1, int s2) {
  std::vector<float> out(indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    const int offset = indices[i][0] * s0 + indices[i][1] * s1 + indices[i][2] * s2;
    out[i] = storage[offset];
  }
  return out;
}

bool run_case(const std::string& name,
              const std::vector<float>& storage,
              const std::vector<std::array<int, 3>>& indices,
              int rank,
              int d0, int d1, int d2,
              int s0, int s1, int s2) {
  const std::vector<float> expected = cpu_reference(storage, indices, rank, s0, s1, s2);
  std::vector<float> actual(indices.size(), 0.0f);

  float* device_storage = nullptr;
  int* device_indices = nullptr;
  float* device_output = nullptr;

  CUDA_CHECK(cudaMalloc(&device_storage, storage.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_indices, indices.size() * 3 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_output, actual.size() * sizeof(float)));

  std::vector<int> flat_indices(indices.size() * 3);
  for (std::size_t i = 0; i < indices.size(); ++i) {
    flat_indices[i * 3 + 0] = indices[i][0];
    flat_indices[i * 3 + 1] = indices[i][1];
    flat_indices[i * 3 + 2] = indices[i][2];
  }

  CUDA_CHECK(cudaMemcpy(device_storage, storage.data(), storage.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_indices, flat_indices.data(), flat_indices.size() * sizeof(int), cudaMemcpyHostToDevice));

  const int threads = 128;
  const int blocks = static_cast<int>((indices.size() + threads - 1) / threads);
  gather_strided_kernel<<<blocks, threads>>>(device_storage, device_indices, device_output,
                                             static_cast<int>(indices.size()), rank,
                                             d0, d1, d2, s0, s1, s2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), device_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_storage));
  CUDA_CHECK(cudaFree(device_indices));
  CUDA_CHECK(cudaFree(device_output));

  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (std::abs(actual[i] - expected[i]) > 1e-6f) {
      std::cerr << name << " failed at index " << i << ": expected " << expected[i]
                << ", got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  const bool ok =
      run_case("rank-three offsets",
               std::vector<float>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                  12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},
               std::vector<std::array<int, 3>>{{0, 1, 2}, {1, 0, 0}, {1, 2, 3}},
               3, 2, 3, 4, 12, 4, 1) &&
      run_case("matrix corners via padded rank-3",
               std::vector<float>{0, 1, 2, 3, 4, 5},
               std::vector<std::array<int, 3>>{{0, 0, 0}, {0, 2, 0}, {1, 0, 0}, {1, 2, 0}},
               2, 2, 3, 1, 3, 1, 1);

  if (!ok) return 1;
  std::cout << "p002 CUDA canonical solution passed stride-gather validation\n";
  return 0;
}
