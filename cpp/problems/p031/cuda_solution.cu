#include "cuda_check.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__global__ void pack_int4_kernel(const std::int8_t* input, std::uint8_t* packed,
                                 int count) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  const int low_index = pair * 2;
  if (low_index >= count) return;

  const std::uint8_t low = static_cast<std::uint8_t>(input[low_index]) & 0x0f;
  std::uint8_t high = 0;
  if (low_index + 1 < count)
    high = (static_cast<std::uint8_t>(input[low_index + 1]) & 0x0f) << 4;
  packed[pair] = static_cast<std::uint8_t>(low | high);
}

__global__ void unpack_int4_kernel(const std::uint8_t* packed, std::int8_t* output,
                                   int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;

  const std::uint8_t byte = packed[index / 2];
  std::uint8_t nibble = (index % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f);
  output[index] = (nibble >= 8) ? static_cast<std::int8_t>(nibble - 16) : static_cast<std::int8_t>(nibble);
}

}  // namespace

int main() {
  const std::vector<std::int8_t> values{-8, -7, -1, 0, 1, 7, 3};
  constexpr int count = 7;
  constexpr int packed_count = 4;

  std::int8_t* d_values = nullptr;
  std::uint8_t* d_packed = nullptr;
  std::int8_t* d_unpacked = nullptr;
  CUDA_CHECK(cudaMalloc(&d_values, count * sizeof(std::int8_t)));
  CUDA_CHECK(cudaMalloc(&d_packed, packed_count * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_unpacked, count * sizeof(std::int8_t)));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), count * sizeof(std::int8_t), cudaMemcpyHostToDevice));

  pack_int4_kernel<<<1, 64>>>(d_values, d_packed, count);
  CUDA_CHECK(cudaGetLastError());
  unpack_int4_kernel<<<1, 64>>>(d_packed, d_unpacked, count);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::array<std::uint8_t, packed_count> packed{};
  std::array<std::int8_t, count> unpacked{};
  CUDA_CHECK(cudaMemcpy(packed.data(), d_packed, packed_count * sizeof(std::uint8_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(unpacked.data(), d_unpacked, count * sizeof(std::int8_t), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_values)); CUDA_CHECK(cudaFree(d_packed)); CUDA_CHECK(cudaFree(d_unpacked));

  const std::array<std::uint8_t, packed_count> expected_bytes{0x98, 0x0f, 0x71, 0x03};
  if (packed != expected_bytes) return 1;
  for (int i = 0; i < count; ++i)
    if (unpacked[i] != values[i]) return 1;

  std::cout << "p031 CUDA int4 pack/unpack passed\n";
  return 0;
}
