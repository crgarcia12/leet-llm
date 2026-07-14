#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void embedding_lookup_kernel(const float* table, const int* token_ids,
                                        float* embeddings, int sequence, int dimension) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = sequence * dimension;
  if (index >= total) return;
  const int token_index = index / dimension;
  const int feature = index % dimension;
  const int token = token_ids[token_index];
  embeddings[index] = table[token * dimension + feature];
}

__global__ void tied_unembedding_kernel(const float* embeddings, const float* table,
                                        float* logits, int sequence, int vocab,
                                        int dimension) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = sequence * vocab;
  if (index >= total) return;
  const int token_index = index / vocab;
  const int vocab_row = index % vocab;

  float sum = 0.0f;
  for (int d = 0; d < dimension; ++d)
    sum += embeddings[token_index * dimension + d] * table[vocab_row * dimension + d];
  logits[index] = sum;
}

void cpu_reference(const std::vector<float>& table, const std::vector<int>& token_ids,
                   int vocab, int dimension, std::vector<float>& embeddings,
                   std::vector<float>& logits) {
  const int sequence = static_cast<int>(token_ids.size());
  embeddings.assign(sequence * dimension, 0.0f);
  logits.assign(sequence * vocab, 0.0f);

  for (int s = 0; s < sequence; ++s) {
    const int token = token_ids[s];
    for (int d = 0; d < dimension; ++d)
      embeddings[s * dimension + d] = table[token * dimension + d];
  }

  for (int s = 0; s < sequence; ++s)
    for (int v = 0; v < vocab; ++v)
      for (int d = 0; d < dimension; ++d)
        logits[s * vocab + v] += embeddings[s * dimension + d] * table[v * dimension + d];
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
  constexpr int vocab = 4;
  constexpr int dimension = 3;
  const std::vector<float> table{
      1.0f, 2.0f, -1.0f,
      0.5f, -0.25f, 3.0f,
      -2.0f, 1.5f, 0.75f,
      4.0f, -1.0f, 2.0f};
  const std::vector<int> token_ids{2, 0, 2, 3};
  const int sequence = static_cast<int>(token_ids.size());

  std::vector<float> expected_embeddings, expected_logits;
  cpu_reference(table, token_ids, vocab, dimension, expected_embeddings, expected_logits);

  std::vector<float> actual_embeddings(sequence * dimension, 0.0f);
  std::vector<float> actual_logits(sequence * vocab, 0.0f);

  float *d_table = nullptr, *d_embeddings = nullptr, *d_logits = nullptr;
  int* d_token_ids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_table, table.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_ids, token_ids.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_embeddings, actual_embeddings.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_logits, actual_logits.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_token_ids, token_ids.data(), token_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

  const int embedding_blocks = (sequence * dimension + kThreads - 1) / kThreads;
  embedding_lookup_kernel<<<embedding_blocks, kThreads>>>(d_table, d_token_ids, d_embeddings,
                                                          sequence, dimension);
  CUDA_CHECK(cudaGetLastError());

  const int logit_blocks = (sequence * vocab + kThreads - 1) / kThreads;
  tied_unembedding_kernel<<<logit_blocks, kThreads>>>(d_embeddings, d_table, d_logits,
                                                      sequence, vocab, dimension);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual_embeddings.data(), d_embeddings,
                        actual_embeddings.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual_logits.data(), d_logits,
                        actual_logits.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_token_ids));
  CUDA_CHECK(cudaFree(d_embeddings));
  CUDA_CHECK(cudaFree(d_logits));

  if (!validate_close(expected_embeddings, actual_embeddings, 2e-5f, "embedding") ||
      !validate_close(expected_logits, actual_logits, 4e-5f, "logit"))
    return 1;

  std::cout << "p013 CUDA canonical solution passed embedding and tied-logit validation\n";
  return 0;
}
