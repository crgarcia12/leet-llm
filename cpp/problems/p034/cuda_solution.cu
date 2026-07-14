#include "cuda_check.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

__global__ void tanh_layer_float_kernel(const float* state, const float* weights,
                                        float* next, int dim) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= dim) return;
  float sum = 0.0f;
  for (int col = 0; col < dim; ++col)
    sum += weights[row * dim + col] * state[col];
  next[row] = tanhf(sum);
}

__global__ void tanh_layer_q4_kernel(const float* state, const std::uint8_t* packed,
                                     const float* scales, float* next,
                                     int dim, int group_size, int groups_per_row,
                                     bool high_nibble_first) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= dim) return;
  float sum = 0.0f;
  for (int col = 0; col < dim; ++col) {
    const int logical = row * dim + col;
    const std::uint8_t byte = packed[logical / 2];
    const std::uint8_t nibble = high_nibble_first
        ? ((logical % 2 == 0) ? ((byte >> 4) & 0x0f) : (byte & 0x0f))
        : ((logical % 2 == 0) ? (byte & 0x0f) : ((byte >> 4) & 0x0f));
    const int q = nibble >= 8 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
    const float w = static_cast<float>(q) * scales[row * groups_per_row + col / group_size];
    sum += w * state[col];
  }
  next[row] = tanhf(sum);
}

int argmax(const std::vector<float>& v) {
  int best = 0;
  for (int i = 1; i < static_cast<int>(v.size()); ++i)
    if (v[i] > v[best]) best = i;
  return best;
}

}  // namespace

int main() {
  constexpr int dim = 5;
  constexpr int layers = 3;
  constexpr int group_size = 3;
  constexpr int groups_per_row = 2;
  constexpr int weight_values = dim * dim;
  constexpr int packed_values = (weight_values + 1) / 2;

  const std::vector<float> initial{0.5f, -0.25f, 0.1f, 0.2f, -0.4f};
  std::vector<float> float_state = initial;
  std::vector<float> q4_state = initial;
  std::vector<float> bad_state = initial;

  std::vector<float> weights(weight_values);
  std::vector<std::uint8_t> packed(packed_values);
  std::vector<float> scales(dim * groups_per_row, 0.1f);
  for (int i = 0; i < weight_values; ++i) {
    const int r = i / dim;
    const int c = i % dim;
    weights[i] = (r == c) ? 0.6f : (0.05f * ((i % 3) - 1));
    const int q = static_cast<int>(std::round(weights[i] / scales[r * groups_per_row + c / group_size]));
    const int clamped = std::max(-8, std::min(7, q));
    const std::uint8_t nibble = static_cast<std::uint8_t>(clamped & 0x0f);
    if ((i & 1) == 0) packed[i / 2] = nibble;
    else packed[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
  }

  float *d_state = nullptr, *d_next = nullptr, *d_weights = nullptr;
  float *d_qstate = nullptr, *d_qnext = nullptr;
  float *d_bstate = nullptr, *d_bnext = nullptr;
  std::uint8_t* d_packed = nullptr;
  float* d_scales = nullptr;
  CUDA_CHECK(cudaMalloc(&d_state, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_next, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, weights.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qstate, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qnext, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bstate, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bnext, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_packed, packed.size() * sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_scales, scales.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_state, float_state.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_qstate, q4_state.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_bstate, bad_state.data(), dim * sizeof(float), cudaMemcpyHostToDevice));

  int bad_first_divergence = -1;
  for (int layer = 0; layer < layers; ++layer) {
    tanh_layer_float_kernel<<<1, 64>>>(d_state, d_weights, d_next, dim);
    CUDA_CHECK(cudaGetLastError());
    tanh_layer_q4_kernel<<<1, 64>>>(d_qstate, d_packed, d_scales, d_qnext, dim, group_size, groups_per_row, false);
    CUDA_CHECK(cudaGetLastError());
    tanh_layer_q4_kernel<<<1, 64>>>(d_bstate, d_packed, d_scales, d_bnext, dim, group_size, groups_per_row, true);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(float_state.data(), d_next, dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(q4_state.data(), d_qnext, dim * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(bad_state.data(), d_bnext, dim * sizeof(float), cudaMemcpyDeviceToHost));

    if (bad_first_divergence < 0 && argmax(float_state) != argmax(bad_state)) bad_first_divergence = layer;

    CUDA_CHECK(cudaMemcpy(d_state, float_state.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_qstate, q4_state.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bstate, bad_state.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
  }

  CUDA_CHECK(cudaFree(d_state)); CUDA_CHECK(cudaFree(d_next)); CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_qstate)); CUDA_CHECK(cudaFree(d_qnext));
  CUDA_CHECK(cudaFree(d_bstate)); CUDA_CHECK(cudaFree(d_bnext));
  CUDA_CHECK(cudaFree(d_packed)); CUDA_CHECK(cudaFree(d_scales));

  if (bad_first_divergence < 0) return 1;
  float rmse = 0.0f;
  for (int i = 0; i < dim; ++i) {
    const float diff = q4_state[i] - float_state[i];
    rmse += diff * diff;
  }
  rmse = std::sqrt(rmse / dim);
  if (!(rmse >= 0.0f && rmse < 0.5f)) return 1;

  std::cout << "p034 CUDA quantization propagation diagnostics passed\n";
  return 0;
}
