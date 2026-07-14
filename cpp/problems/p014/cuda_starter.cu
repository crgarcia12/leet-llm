#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void q_projection_todo_kernel(const float* hidden, const float* wq,
                                         float* q, int sequence, int model_dim,
                                         int q_cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = sequence * q_cols;
  if (index >= total) return;
  const int row = index / q_cols;
  const int col = index % q_cols;

  // TODO(p014): include all model_dim terms and implement K/V projections too.
  q[index] = hidden[row * model_dim] * wq[col];
}

}  // namespace

int main() {
  constexpr int sequence = 1;
  constexpr int model_dim = 2;
  constexpr int q_cols = 2;
  const std::vector<float> hidden{2.0f, 3.0f};
  const std::vector<float> wq{1.0f, 0.0f, 0.0f, 2.0f};
  std::vector<float> q(sequence * q_cols, 0.0f);

  float *d_hidden = nullptr, *d_wq = nullptr, *d_q = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hidden, hidden.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wq, wq.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, q.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wq, wq.data(), wq.size() * sizeof(float), cudaMemcpyHostToDevice));

  q_projection_todo_kernel<<<1, 64>>>(d_hidden, d_wq, d_q, sequence, model_dim, q_cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(q.data(), d_q, q.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_wq));
  CUDA_CHECK(cudaFree(d_q));

  const std::vector<float> expected{2.0f, 6.0f};
  const bool passed = std::abs(q[0] - expected[0]) <= 1e-5f && std::abs(q[1] - expected[1]) <= 1e-5f;
  if (passed) {
    std::cerr << "p014 starter unexpectedly passed; TODO complete Q/K/V projection contract\n";
    return 1;
  }
  std::cerr << "p014 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
