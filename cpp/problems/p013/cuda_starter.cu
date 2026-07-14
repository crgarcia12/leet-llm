#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void embedding_lookup_todo_kernel(const float* table, const int* token_ids,
                                             float* embeddings, int sequence,
                                             int dimension) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = sequence * dimension;
  if (index >= total) return;

  const int token_index = index / dimension;
  const int feature = index % dimension;
  // TODO(p013): copy the full embedding row for each token.
  embeddings[index] = (feature == 0) ? table[token_ids[token_index] * dimension] : 0.0f;
}

}  // namespace

int main() {
  constexpr int vocab = 3;
  constexpr int dimension = 2;
  const std::vector<float> table{1, 2, 3, 4, 5, 6};
  const std::vector<int> token_ids{2, 0};
  std::vector<float> embeddings(token_ids.size() * dimension, 0.0f);

  float* d_table = nullptr;
  int* d_token_ids = nullptr;
  float* d_embeddings = nullptr;
  CUDA_CHECK(cudaMalloc(&d_table, table.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_token_ids, token_ids.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_embeddings, embeddings.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_token_ids, token_ids.data(), token_ids.size() * sizeof(int), cudaMemcpyHostToDevice));

  embedding_lookup_todo_kernel<<<1, 64>>>(d_table, d_token_ids, d_embeddings,
                                          static_cast<int>(token_ids.size()), dimension);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(embeddings.data(), d_embeddings,
                        embeddings.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_table));
  CUDA_CHECK(cudaFree(d_token_ids));
  CUDA_CHECK(cudaFree(d_embeddings));

  const std::vector<float> expected{5, 6, 1, 2};
  const bool passed = std::abs(embeddings[0] - expected[0]) <= 1e-5f &&
                      std::abs(embeddings[1] - expected[1]) <= 1e-5f &&
                      std::abs(embeddings[2] - expected[2]) <= 1e-5f &&
                      std::abs(embeddings[3] - expected[3]) <= 1e-5f;
  if (passed) {
    std::cerr << "p013 starter unexpectedly passed; TODO gather and tied logits are incomplete\n";
    return 1;
  }
  std::cerr << "p013 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
