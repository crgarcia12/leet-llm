#include "cuda_check.hpp"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void to_token_major_kernel(const float* logical, float* token_major,
                                      int layers, int tokens, int heads, int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layers * tokens * heads * dim;
  if (index >= count) return;

  const int d = index % dim;
  const int h = (index / dim) % heads;
  const int t = (index / (dim * heads)) % tokens;
  const int l = index / (dim * heads * tokens);

  const int logical_index = (((l * tokens + t) * heads + h) * dim + d);
  const int token_index = (((l * tokens + t) * heads + h) * dim + d);
  token_major[token_index] = logical[logical_index];
}

__global__ void to_head_major_kernel(const float* logical, float* head_major,
                                     int layers, int tokens, int heads, int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layers * tokens * heads * dim;
  if (index >= count) return;

  const int d = index % dim;
  const int h = (index / dim) % heads;
  const int t = (index / (dim * heads)) % tokens;
  const int l = index / (dim * heads * tokens);

  const int logical_index = (((l * tokens + t) * heads + h) * dim + d);
  const int head_index = (((l * heads + h) * tokens + t) * dim + d);
  head_major[head_index] = logical[logical_index];
}

__global__ void from_head_major_kernel(const float* head_major, float* logical,
                                       int layers, int tokens, int heads, int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int count = layers * tokens * heads * dim;
  if (index >= count) return;

  const int d = index % dim;
  const int h = (index / dim) % heads;
  const int t = (index / (dim * heads)) % tokens;
  const int l = index / (dim * heads * tokens);

  const int head_index = (((l * heads + h) * tokens + t) * dim + d);
  const int logical_index = (((l * tokens + t) * heads + h) * dim + d);
  logical[logical_index] = head_major[head_index];
}

std::size_t count_spans(const std::vector<int>& offsets) {
  if (offsets.empty()) return 0;
  std::size_t spans = 1;
  for (std::size_t i = 1; i < offsets.size(); ++i)
    if (offsets[i] != offsets[i - 1] + 1) ++spans;
  return spans;
}

bool validate_equal(const std::vector<float>& expected, const std::vector<float>& actual,
                    const char* label) {
  for (std::size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] != actual[i]) {
      std::cerr << label << " mismatch at index " << i << " expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  constexpr int layers = 2;
  constexpr int tokens = 4;
  constexpr int heads = 2;
  constexpr int dim = 3;
  const int count = layers * tokens * heads * dim;

  std::vector<float> logical(count, 0.0f);
  for (int i = 0; i < count; ++i) logical[i] = static_cast<float>(i);

  std::vector<float> token_major(count, -1.0f);
  std::vector<float> head_major(count, -1.0f);
  std::vector<float> roundtrip(count, -2.0f);

  float *d_logical = nullptr, *d_token = nullptr, *d_head = nullptr, *d_roundtrip = nullptr;
  CUDA_CHECK(cudaMalloc(&d_logical, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_head, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_roundtrip, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_logical, logical.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  to_token_major_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
      d_logical, d_token, layers, tokens, heads, dim);
  CUDA_CHECK(cudaGetLastError());
  to_head_major_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
      d_logical, d_head, layers, tokens, heads, dim);
  CUDA_CHECK(cudaGetLastError());
  from_head_major_kernel<<<(count + kThreads - 1) / kThreads, kThreads>>>(
      d_head, d_roundtrip, layers, tokens, heads, dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(token_major.data(), d_token, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(head_major.data(), d_head, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(roundtrip.data(), d_roundtrip, count * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_logical));
  CUDA_CHECK(cudaFree(d_token));
  CUDA_CHECK(cudaFree(d_head));
  CUDA_CHECK(cudaFree(d_roundtrip));

  const std::size_t token_offset = (((1 * tokens + 2) * heads + 1) * dim + 2);
  const std::size_t head_offset = (((1 * heads + 1) * tokens + 2) * dim + 2);

  std::vector<int> token_trace;
  std::vector<int> head_trace;
  for (int t = 0; t < tokens; ++t)
    for (int d = 0; d < dim; ++d) {
      token_trace.push_back(static_cast<int>((((1 * tokens + t) * heads + 1) * dim + d)));
      head_trace.push_back(static_cast<int>((((1 * heads + 1) * tokens + t) * dim + d)));
    }

  const bool ok = validate_equal(logical, token_major, "token-major") &&
                  validate_equal(logical, roundtrip, "round-trip") &&
                  token_offset == 41 && head_offset == 44 &&
                  count_spans(token_trace) == 4 && count_spans(head_trace) == 1;

  if (!ok) return 1;

  std::cout << "p024 CUDA canonical solution passed KV layout offset and round-trip validation\n";
  return 0;
}
