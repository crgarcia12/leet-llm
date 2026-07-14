#include "cuda_check.hpp"

#include <iostream>
#include <vector>

__global__ void gather_embeddings_kernel(const int* token_ids, const float* table,
                                         float* residual, int sequence, int dim) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= sequence * dim) return;
  const int token = index / dim;
  const int feature = index % dim;
  residual[index] = table[token_ids[token] * dim + feature];
}

int main() {
  constexpr int vocab = 7, dim = 4;
  const std::vector<int> prompt{1, 4, 2};
  std::vector<float> embedding(vocab * dim, 0.1f), residual(prompt.size() * dim);

  int* d_tokens = nullptr;
  float *d_embedding = nullptr, *d_residual = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tokens, prompt.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_embedding, embedding.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_residual, residual.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_tokens, prompt.data(), prompt.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_embedding, embedding.data(), embedding.size() * sizeof(float), cudaMemcpyHostToDevice));
  gather_embeddings_kernel<<<(static_cast<int>(residual.size()) + 127) / 128, 128>>>(
      d_tokens, d_embedding, d_residual, static_cast<int>(prompt.size()), dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(residual.data(), d_residual, residual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_tokens));
  CUDA_CHECK(cudaFree(d_embedding));
  CUDA_CHECK(cudaFree(d_residual));

  // TODO: project gathered residual rows to K/V and append absolute-position cache entries.
  std::cout << "p039 starter builds. TODO: KV projection/cache append path is incomplete.\n";
  return 0;
}
