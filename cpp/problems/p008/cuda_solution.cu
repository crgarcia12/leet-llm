#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

__global__ void gemv_rows_kernel(const float* matrix, const float* vector, float* output,
                                 int rows, int columns) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int c = 0; c < columns; ++c)
    sum += matrix[row * columns + c] * vector[c];
  output[row] = sum;
}

__global__ void swiglu_gate_kernel(const float* gate, const float* up, float* hidden, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float g = gate[index];
  hidden[index] = (g / (1.0f + expf(-g))) * up[index];
}

std::vector<float> cpu_reference(const std::vector<float>& input,
                                 const std::vector<float>& gate_w,
                                 const std::vector<float>& up_w,
                                 const std::vector<float>& down_w,
                                 int d, int h, int o) {
  std::vector<float> gate(h, 0.0f), up(h, 0.0f), hidden(h, 0.0f), output(o, 0.0f);
  for (int row = 0; row < h; ++row) {
    for (int col = 0; col < d; ++col) {
      gate[row] += gate_w[row * d + col] * input[col];
      up[row] += up_w[row * d + col] * input[col];
    }
    hidden[row] = (gate[row] / (1.0f + std::exp(-gate[row]))) * up[row];
  }
  for (int row = 0; row < o; ++row)
    for (int col = 0; col < h; ++col)
      output[row] += down_w[row * h + col] * hidden[col];
  return output;
}

bool run_case() {
  const int d = 2, h = 3, o = 2;
  const std::vector<float> input{1, -2};
  const std::vector<float> gate_w{1, 0, 0, 1, 1, -1};
  const std::vector<float> up_w{2, 1, -1, 1, 0.5f, 2};
  const std::vector<float> down_w{1, 0.5f, -1, -0.25f, 2, 0.75f};

  const std::vector<float> expected = cpu_reference(input, gate_w, up_w, down_w, d, h, o);
  std::vector<float> actual(o, 0.0f);

  float *d_input = nullptr, *d_gate_w = nullptr, *d_up_w = nullptr, *d_down_w = nullptr;
  float *d_gate = nullptr, *d_up = nullptr, *d_hidden = nullptr, *d_output = nullptr;

  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gate_w, gate_w.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up_w, up_w.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_down_w, down_w.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_gate, h * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_up, h * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_hidden, h * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, o * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_gate_w, gate_w.data(), gate_w.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_up_w, up_w.data(), up_w.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_down_w, down_w.data(), down_w.size() * sizeof(float), cudaMemcpyHostToDevice));

  gemv_rows_kernel<<<1, 128>>>(d_gate_w, d_input, d_gate, h, d);
  gemv_rows_kernel<<<1, 128>>>(d_up_w, d_input, d_up, h, d);
  swiglu_gate_kernel<<<1, 128>>>(d_gate, d_up, d_hidden, h);
  gemv_rows_kernel<<<1, 128>>>(d_down_w, d_hidden, d_output, o, h);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(actual.data(), d_output, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gate_w));
  CUDA_CHECK(cudaFree(d_up_w));
  CUDA_CHECK(cudaFree(d_down_w));
  CUDA_CHECK(cudaFree(d_gate));
  CUDA_CHECK(cudaFree(d_up));
  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_output));

  for (int i = 0; i < o; ++i) {
    const float scale = std::max({1.0f, std::abs(expected[i]), std::abs(actual[i])});
    if (std::abs(expected[i] - actual[i]) > 4e-5f * scale) {
      std::cerr << "SwiGLU mismatch at output " << i << ": expected " << expected[i]
                << " got " << actual[i] << '\n';
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  if (!run_case()) return 1;
  std::cout << "p008 CUDA canonical solution passed SwiGLU gate/up/down validation\n";
  return 0;
}
