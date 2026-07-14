#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void rope_kernel(const float* input, float* output,
                            int sequence, int heads, int head_dim,
                            int rotary_dim, int position_offset, float base) {
  const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_pairs = sequence * heads * (rotary_dim / 2);
  if (pair_index >= total_pairs) return;

  const int pairs_per_row = rotary_dim / 2;
  const int row = pair_index / pairs_per_row;
  const int pair = pair_index % pairs_per_row;
  const int sequence_index = row / heads;
  const int head = row % heads;
  const int feature = pair * 2;
  const int base_index = (sequence_index * heads + head) * head_dim;
  const float x = input[base_index + feature];
  const float y = input[base_index + feature + 1];

  const float position = static_cast<float>(position_offset + sequence_index);
  const float exponent = static_cast<float>(feature) / static_cast<float>(rotary_dim);
  const float theta = position / powf(base, exponent);
  const float c = cosf(theta);
  const float s = sinf(theta);

  output[base_index + feature] = x * c - y * s;
  output[base_index + feature + 1] = x * s + y * c;
}

std::vector<float> cpu_rope(const std::vector<float>& input, int sequence, int heads,
                            int head_dim, int rotary_dim, int position_offset,
                            float base) {
  std::vector<float> output(input);
  for (int s = 0; s < sequence; ++s) {
    const float position = static_cast<float>(position_offset + s);
    for (int h = 0; h < heads; ++h) {
      const int row = (s * heads + h) * head_dim;
      for (int feature = 0; feature < rotary_dim; feature += 2) {
        const float exponent = static_cast<float>(feature) / static_cast<float>(rotary_dim);
        const float theta = position / std::pow(base, exponent);
        const float c = std::cos(theta);
        const float s_val = std::sin(theta);
        const float x = input[row + feature];
        const float y = input[row + feature + 1];
        output[row + feature] = x * c - y * s_val;
        output[row + feature + 1] = x * s_val + y * c;
      }
    }
  }
  return output;
}

bool validate_close(const std::vector<float>& expected, const std::vector<float>& actual,
                    float tolerance, const char* label) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float scale = std::max({1.0f, std::abs(expected[i]), std::abs(actual[i])});
    if (std::abs(expected[i] - actual[i]) > tolerance * scale) {
      std::cerr << label << " mismatch at index " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int query_sequence = 2;
  constexpr int key_sequence = 3;
  constexpr int query_heads = 1;
  constexpr int key_heads = 1;
  constexpr int head_dim = 6;
  constexpr int rotary_dim = 4;
  constexpr float base = 100.0f;

  const std::vector<float> queries{
      1, 0, 0, 2, 9, 10,
      -1, 2, 3, -4, 5, -6};
  const std::vector<float> keys{
      0.5f, -1, 2, 1, 7, 8,
      1.5f, 0.25f, -2, 3, 4, 5,
      -3, 1, 0.5f, -0.5f, 6, 7};

  const std::vector<float> expected_q =
      cpu_rope(queries, query_sequence, query_heads, head_dim, rotary_dim, 5, base);
  const std::vector<float> expected_k =
      cpu_rope(keys, key_sequence, key_heads, head_dim, rotary_dim, 3, base);

  std::vector<float> actual_q(queries);
  std::vector<float> actual_k(keys);

  float *d_q_in = nullptr, *d_q_out = nullptr, *d_k_in = nullptr, *d_k_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_q_in, queries.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q_out, queries.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_in, keys.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k_out, keys.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q_in, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q_out, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_in, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_out, keys.data(), keys.size() * sizeof(float), cudaMemcpyHostToDevice));

  rope_kernel<<<(query_sequence * query_heads * (rotary_dim / 2) + kThreads - 1) / kThreads, kThreads>>>(
      d_q_in, d_q_out, query_sequence, query_heads, head_dim, rotary_dim, 5, base);
  CUDA_CHECK(cudaGetLastError());
  rope_kernel<<<(key_sequence * key_heads * (rotary_dim / 2) + kThreads - 1) / kThreads, kThreads>>>(
      d_k_in, d_k_out, key_sequence, key_heads, head_dim, rotary_dim, 3, base);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual_q.data(), d_q_out, actual_q.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual_k.data(), d_k_out, actual_k.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_q_in));
  CUDA_CHECK(cudaFree(d_q_out));
  CUDA_CHECK(cudaFree(d_k_in));
  CUDA_CHECK(cudaFree(d_k_out));

  if (!validate_close(expected_q, actual_q, 8e-5f, "Q") ||
      !validate_close(expected_k, actual_k, 8e-5f, "K"))
    return 1;

  std::cout << "p015 CUDA canonical solution passed RoPE rotation validation\n";
  return 0;
}
