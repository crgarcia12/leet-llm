#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void tiled_attention_todo_kernel(const float* scores,
                                            const float* values,
                                            float* output,
                                            int count,
                                            int tile_size) {
  // TODO(p020): apply one recurrence merge per tile, not per score.
  float running_max = -CUDART_INF_F;
  float running_denominator = 0.0f;
  float running_output = 0.0f;

  for (int tile_start = 0; tile_start < count; tile_start += tile_size) {
    const int tile_end = min(tile_start + tile_size, count);
    for (int i = tile_start; i < tile_end; ++i) {
      const float next_max = fmaxf(running_max, scores[i]);
      const float alpha = (running_max == -CUDART_INF_F) ? 0.0f : expf(running_max - next_max);
      const float beta = expf(scores[i] - next_max);
      running_denominator = running_denominator * alpha + beta;
      running_output = running_output * alpha + beta * values[i];
      running_max = next_max;
    }
  }
  output[0] = running_output / running_denominator;
}

}  // namespace

int main() {
  const std::vector<float> scores{1.0f, 3.0f};
  const std::vector<float> values{10.0f, 20.0f};
  std::vector<float> output(1, 0.0f);

  float *d_scores = nullptr, *d_values = nullptr, *d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_scores, scores.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_values, values.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_scores, scores.data(), scores.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_values, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));

  tiled_attention_todo_kernel<<<1, 1>>>(d_scores, d_values, d_output, static_cast<int>(scores.size()), 2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_scores));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_output));

  const float expected = 19.046377f;  // tile-wise merge with prior state example differs.
  if (std::abs(output[0] - expected) <= 1e-3f) {
    std::cerr << "p020 starter unexpectedly passed; TODO tile merge semantics are incomplete\n";
    return 1;
  }
  std::cerr << "p020 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
