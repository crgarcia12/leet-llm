#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 16;

__global__ void gemm_tiled_kernel(const float* lhs, const float* rhs, float* output,
                                  int m, int k, int n) {
  __shared__ float lhs_tile[kTile][kTile];
  __shared__ float rhs_tile[kTile][kTile];

  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int row = blockIdx.y * kTile + local_y;
  const int col = blockIdx.x * kTile + local_x;

  float acc = 0.0f;
  const int k_tiles = (k + kTile - 1) / kTile;
  for (int tile_idx = 0; tile_idx < k_tiles; ++tile_idx) {
    const int lhs_col = tile_idx * kTile + local_x;
    const int rhs_row = tile_idx * kTile + local_y;

    lhs_tile[local_y][local_x] = (row < m && lhs_col < k) ? lhs[row * k + lhs_col] : 0.0f;
    rhs_tile[local_y][local_x] = (rhs_row < k && col < n) ? rhs[rhs_row * n + col] : 0.0f;
    __syncthreads();

    for (int inner = 0; inner < kTile; ++inner)
      acc += lhs_tile[local_y][inner] * rhs_tile[inner][local_x];
    __syncthreads();
  }

  if (row < m && col < n) output[row * n + col] = acc;
}

std::vector<float> cpu_reference(const std::vector<float>& lhs, const std::vector<float>& rhs,
                                 int m, int k, int n) {
  std::vector<float> out(m * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      double sum = 0.0;
      for (int inner = 0; inner < k; ++inner)
        sum += static_cast<double>(lhs[row * k + inner]) * rhs[inner * n + col];
      out[row * n + col] = static_cast<float>(sum);
    }
  }
  return out;
}

bool run_case(const std::string& name, const std::vector<float>& lhs, const std::vector<float>& rhs,
              int m, int k, int n, float tolerance) {
  std::vector<float> expected = cpu_reference(lhs, rhs, m, k, n);
  std::vector<float> actual(expected.size(), 0.0f);
  if (m == 0 || n == 0) return true;

  float *device_lhs = nullptr, *device_rhs = nullptr, *device_out = nullptr;
  CUDA_CHECK(cudaMalloc(&device_lhs, lhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_rhs, rhs.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_out, actual.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_lhs, lhs.data(), lhs.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_rhs, rhs.data(), rhs.size() * sizeof(float), cudaMemcpyHostToDevice));

  if (k == 0) {
    CUDA_CHECK(cudaMemset(device_out, 0, actual.size() * sizeof(float)));
  } else {
    dim3 threads(kTile, kTile);
    dim3 blocks((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    gemm_tiled_kernel<<<blocks, threads>>>(device_lhs, device_rhs, device_out, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  CUDA_CHECK(cudaMemcpy(actual.data(), device_out, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_lhs));
  CUDA_CHECK(cudaFree(device_rhs));
  CUDA_CHECK(cudaFree(device_out));

  for (std::size_t i = 0; i < actual.size(); ++i) {
    const float scale = std::max({1.0f, std::abs(expected[i]), std::abs(actual[i])});
    if (std::abs(expected[i] - actual[i]) > tolerance * scale) {
      std::cerr << name << " failed at index " << i << " expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  std::vector<float> edge_lhs(17 * 19), edge_rhs(19 * 18);
  for (int i = 0; i < static_cast<int>(edge_lhs.size()); ++i)
    edge_lhs[i] = static_cast<float>((i % 23) - 11) / 12.0f;
  for (int i = 0; i < static_cast<int>(edge_rhs.size()); ++i)
    edge_rhs[i] = static_cast<float>((i % 31) - 15) / 16.0f;

  const bool ok =
      run_case("small rectangular product", {1, 2, 3, 4, 5, 6}, {1, 2, 0, -1, 3, 0}, 2, 3, 2, 4e-5f) &&
      run_case("zero inner dimension", {}, {}, 2, 0, 3, 4e-5f) &&
      run_case("partial tiles in every dimension", edge_lhs, edge_rhs, 17, 19, 18, 4e-4f);

  if (!ok) return 1;
  std::cout << "p005 CUDA canonical solution passed GEMM validation\n";
  return 0;
}
