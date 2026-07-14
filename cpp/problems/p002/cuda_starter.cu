#include "cuda_check.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void gather_strided_kernel_todo(const float* storage, const int* indices, float* output,
                                           int count, int rank,
                                           int d0, int d1, int d2,
                                           int s0, int s1, int s2) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  // TODO(p002): map logical indices through strides to a flat offset.
  // Placeholder always reads the first element.
  output[tid] = storage[0];
}

float cpu_reference_value(const std::vector<float>& storage, const std::array<int, 3>& index,
                          int s0, int s1, int s2) {
  return storage[index[0] * s0 + index[1] * s1 + index[2] * s2];
}

}  // namespace

int main() {
  const std::vector<float> storage{0, 1, 2, 3, 4, 5};
  const std::vector<std::array<int, 3>> indices{{0, 0, 0}, {1, 2, 0}};
  std::vector<int> flat{0, 0, 0, 1, 2, 0};
  std::vector<float> output(2, 0.0f);

  float* device_storage = nullptr;
  int* device_indices = nullptr;
  float* device_output = nullptr;
  CUDA_CHECK(cudaMalloc(&device_storage, storage.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_indices, flat.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&device_output, output.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(device_storage, storage.data(), storage.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_indices, flat.data(), flat.size() * sizeof(int), cudaMemcpyHostToDevice));

  gather_strided_kernel_todo<<<1, 64>>>(device_storage, device_indices, device_output, 2, 2, 2, 3, 1, 3, 1, 1);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(output.data(), device_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_storage));
  CUDA_CHECK(cudaFree(device_indices));
  CUDA_CHECK(cudaFree(device_output));

  const float expected_last = cpu_reference_value(storage, indices[1], 3, 1, 1);
  if (std::abs(output[1] - expected_last) <= 1e-6f) {
    std::cerr << "p002 starter unexpectedly passed; TODO stride mapping is still missing\n";
    return 1;
  }
  std::cerr << "p002 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
