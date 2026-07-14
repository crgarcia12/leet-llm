#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kTile = 16;

__global__ void transpose_tiled_kernel(const float* input, float* output, int rows, int columns) {
  __shared__ float tile[kTile][kTile + 1];

  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  const int input_col = blockIdx.x * kTile + local_x;
  const int input_row = blockIdx.y * kTile + local_y;

  if (input_row < rows && input_col < columns) {
    tile[local_y][local_x] = input[input_row * columns + input_col];
  }
  __syncthreads();

  const int output_col = blockIdx.y * kTile + local_x;
  const int output_row = blockIdx.x * kTile + local_y;
  if (output_row < columns && output_col < rows) {
    output[output_row * rows + output_col] = tile[local_x][local_y];
  }
}

std::vector<float> cpu_transpose(const std::vector<float>& input, int rows, int columns) {
  std::vector<float> output(input.size());
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < columns; ++c)
      output[c * rows + r] = input[r * columns + c];
  return output;
}

bool run_case(const std::string& name, const std::vector<float>& input, int rows, int columns) {
  std::vector<float> expected = cpu_transpose(input, rows, columns);
  std::vector<float> actual(expected.size(), 0.0f);

  float *device_input = nullptr, *device_output = nullptr;
  const std::size_t bytes = input.size() * sizeof(float);
  if (!input.empty()) {
    CUDA_CHECK(cudaMalloc(&device_input, bytes));
    CUDA_CHECK(cudaMalloc(&device_output, bytes));
    CUDA_CHECK(cudaMemcpy(device_input, input.data(), bytes, cudaMemcpyHostToDevice));

    dim3 threads(kTile, kTile);
    dim3 blocks((columns + kTile - 1) / kTile, (rows + kTile - 1) / kTile);
    transpose_tiled_kernel<<<blocks, threads>>>(device_input, device_output, rows, columns);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(actual.data(), device_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));
  }

  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (std::abs(expected[i] - actual[i]) > 1e-6f) {
      std::cerr << name << " failed at index " << i << " expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  std::vector<float> edge(17 * 19);
  for (int i = 0; i < static_cast<int>(edge.size()); ++i) edge[i] = static_cast<float>(i - 100) / 7.0f;

  const bool ok =
      run_case("rectangular matrix", {1, 2, 3, 4, 5, 6}, 2, 3) &&
      run_case("single row", {-2, 0, 7}, 1, 3) &&
      run_case("empty rows", {}, 0, 5) &&
      run_case("crosses tile edges", edge, 17, 19);

  if (!ok) return 1;
  std::cout << "p003 CUDA canonical solution passed tiled transpose validation\n";
  return 0;
}
