#include "cuda_check.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

__global__ void decode_u32_le_kernel(const std::uint8_t* bytes, std::uint32_t* out, int offset) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  const int o = offset;
  out[0] = static_cast<std::uint32_t>(bytes[o]) |
           (static_cast<std::uint32_t>(bytes[o + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[o + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[o + 3]) << 24);
}

__global__ void decode_f32_le_kernel(const std::uint8_t* bytes, float* out, int float_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= float_count) return;
  const int o = idx * 4;
  const std::uint32_t bits = static_cast<std::uint32_t>(bytes[o]) |
                             (static_cast<std::uint32_t>(bytes[o + 1]) << 8) |
                             (static_cast<std::uint32_t>(bytes[o + 2]) << 16) |
                             (static_cast<std::uint32_t>(bytes[o + 3]) << 24);
  out[idx] = __uint_as_float(bits);
}

}  // namespace

int main() {
  const std::array<std::uint8_t, 8> magic{'L','L','M','W','G','T','0','1'};
  const std::uint32_t version = 1;
  const std::uint64_t header_length = 24;
  const std::vector<std::uint8_t> payload{
      0x00, 0x00, 0x80, 0x3f,
      0x00, 0x00, 0x20, 0xc0,
      0x00, 0x00, 0x00, 0x3e,
  };

  std::vector<std::uint8_t> file;
  file.insert(file.end(), magic.begin(), magic.end());
  for (int i = 0; i < 4; ++i) file.push_back(static_cast<std::uint8_t>((version >> (8 * i)) & 0xff));
  for (int i = 0; i < 8; ++i) file.push_back(static_cast<std::uint8_t>((header_length >> (8 * i)) & 0xff));
  file.resize(20 + header_length, 0x41);
  while ((file.size() % 8) != 0) file.push_back(0x00);
  const std::size_t payload_offset = file.size();
  file.insert(file.end(), payload.begin(), payload.end());

  std::uint8_t* d_file = nullptr;
  std::uint32_t* d_version = nullptr;
  float* d_values = nullptr;
  CUDA_CHECK(cudaMalloc(&d_file, file.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_version, sizeof(std::uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_values, 3 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_file, file.data(), file.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));

  decode_u32_le_kernel<<<1, 1>>>(d_file, d_version, 8);
  CUDA_CHECK(cudaGetLastError());
  decode_f32_le_kernel<<<1, 64>>>(d_file + payload_offset, d_values, 3);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::uint32_t decoded_version = 0;
  std::vector<float> decoded_values(3);
  CUDA_CHECK(cudaMemcpy(&decoded_version, d_version, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(decoded_values.data(), d_values, 3 * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_file)); CUDA_CHECK(cudaFree(d_version)); CUDA_CHECK(cudaFree(d_values));

  if (decoded_version != 1u) return 1;
  if (std::memcmp(file.data(), magic.data(), magic.size()) != 0) return 1;
  if (decoded_values[0] != 1.0f || decoded_values[1] != -2.5f || decoded_values[2] != 0.125f) return 1;

  std::cout << "p036 CUDA weight-format little-endian decode passed\n";
  return 0;
}
