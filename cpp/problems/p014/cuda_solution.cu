#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr int kThreads = 128;

__global__ void projection_kernel(const float* hidden, const float* weights, float* output,
                                  int sequence, int model_dim, int output_cols) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = sequence * output_cols;
  if (index >= total) return;

  const int row = index / output_cols;
  const int col = index % output_cols;
  float sum = 0.0f;
  for (int d = 0; d < model_dim; ++d)
    sum += hidden[row * model_dim + d] * weights[d * output_cols + col];
  output[index] = sum;
}

std::vector<float> cpu_projection(const std::vector<float>& hidden,
                                  const std::vector<float>& weights,
                                  int sequence, int model_dim, int out_cols) {
  std::vector<float> output(sequence * out_cols, 0.0f);
  for (int s = 0; s < sequence; ++s)
    for (int c = 0; c < out_cols; ++c)
      for (int d = 0; d < model_dim; ++d)
        output[s * out_cols + c] += hidden[s * model_dim + d] * weights[d * out_cols + c];
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
  constexpr int sequence = 2;
  constexpr int model_dim = 3;
  constexpr int query_heads = 2;
  constexpr int kv_heads = 1;
  constexpr int head_dim = 2;
  constexpr int q_cols = query_heads * head_dim;
  constexpr int kv_cols = kv_heads * head_dim;

  const std::vector<float> hidden{
      1.0f, -2.0f, 0.5f,
      0.0f, 1.5f, -1.0f};
  const std::vector<float> wq{
      1.0f, 0.0f, 2.0f, -1.0f,
      0.5f, -1.5f, 0.0f, 1.0f,
      -0.25f, 1.0f, 1.5f, 0.0f};
  const std::vector<float> wk{
      0.5f, -1.0f,
      1.0f, 0.25f,
      -0.5f, 2.0f};
  const std::vector<float> wv{
      1.0f, 0.5f,
      -1.0f, 1.0f,
      0.25f, -0.75f};

  const std::vector<float> expected_q = cpu_projection(hidden, wq, sequence, model_dim, q_cols);
  const std::vector<float> expected_k = cpu_projection(hidden, wk, sequence, model_dim, kv_cols);
  const std::vector<float> expected_v = cpu_projection(hidden, wv, sequence, model_dim, kv_cols);

  std::vector<float> actual_q(expected_q.size(), 0.0f);
  std::vector<float> actual_k(expected_k.size(), 0.0f);
  std::vector<float> actual_v(expected_v.size(), 0.0f);

  float *d_hidden = nullptr, *d_wq = nullptr, *d_wk = nullptr, *d_wv = nullptr;
  float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
  CUDA_CHECK(cudaMalloc(&d_hidden, hidden.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wq, wq.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wk, wk.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_wv, wv.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_q, actual_q.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_k, actual_k.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v, actual_v.size() * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_hidden, hidden.data(), hidden.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wq, wq.data(), wq.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wk, wk.data(), wk.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_wv, wv.data(), wv.size() * sizeof(float), cudaMemcpyHostToDevice));

  projection_kernel<<<(sequence * q_cols + kThreads - 1) / kThreads, kThreads>>>(
      d_hidden, d_wq, d_q, sequence, model_dim, q_cols);
  CUDA_CHECK(cudaGetLastError());
  projection_kernel<<<(sequence * kv_cols + kThreads - 1) / kThreads, kThreads>>>(
      d_hidden, d_wk, d_k, sequence, model_dim, kv_cols);
  CUDA_CHECK(cudaGetLastError());
  projection_kernel<<<(sequence * kv_cols + kThreads - 1) / kThreads, kThreads>>>(
      d_hidden, d_wv, d_v, sequence, model_dim, kv_cols);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual_q.data(), d_q, actual_q.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual_k.data(), d_k, actual_k.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(actual_v.data(), d_v, actual_v.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_wq));
  CUDA_CHECK(cudaFree(d_wk));
  CUDA_CHECK(cudaFree(d_wv));
  CUDA_CHECK(cudaFree(d_q));
  CUDA_CHECK(cudaFree(d_k));
  CUDA_CHECK(cudaFree(d_v));

  if (!validate_close(expected_q, actual_q, 5e-5f, "Q") ||
      !validate_close(expected_k, actual_k, 5e-5f, "K") ||
      !validate_close(expected_v, actual_v, 5e-5f, "V"))
    return 1;

  std::cout << "p014 CUDA canonical solution passed Q/K/V projection validation\n";
  return 0;
}
