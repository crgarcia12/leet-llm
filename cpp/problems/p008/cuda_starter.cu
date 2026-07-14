#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

__global__ void gemv_rows_todo_kernel(const float* matrix, const float* vector, float* output,
                                      int rows, int columns) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    // TODO(p008): compute complete dot product over columns.
    output[row] = matrix[row * columns] * vector[0];
  }
}

__global__ void swiglu_gate_todo_kernel(const float* gate, const float* up, float* hidden, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    // TODO(p008): use SiLU(gate) * up. This is only sigmoid(gate) * up.
    hidden[index] = (1.0f / (1.0f + expf(-gate[index]))) * up[index];
  }
}

}  // namespace

int main() {
  const int d = 2, h = 2, o = 2;
  const std::vector<float> input{1, 2};
  const std::vector<float> gate_w{1, 0, 0, -1};
  const std::vector<float> up_w{0, 1, 1, 1};
  const std::vector<float> down_w{1, 1, 2, -1};

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

  gemv_rows_todo_kernel<<<1, 64>>>(d_gate_w, d_input, d_gate, h, d);
  gemv_rows_todo_kernel<<<1, 64>>>(d_up_w, d_input, d_up, h, d);
  swiglu_gate_todo_kernel<<<1, 64>>>(d_gate, d_up, d_hidden, h);
  gemv_rows_todo_kernel<<<1, 64>>>(d_down_w, d_hidden, d_output, o, h);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> output(o);
  CUDA_CHECK(cudaMemcpy(output.data(), d_output, o * sizeof(float), cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_gate_w));
  CUDA_CHECK(cudaFree(d_up_w));
  CUDA_CHECK(cudaFree(d_down_w));
  CUDA_CHECK(cudaFree(d_gate));
  CUDA_CHECK(cudaFree(d_up));
  CUDA_CHECK(cudaFree(d_hidden));
  CUDA_CHECK(cudaFree(d_output));

  const std::vector<float> expected{0.746900f, 3.639454f};
  const bool passed = std::abs(output[0] - expected[0]) <= 1e-3f && std::abs(output[1] - expected[1]) <= 1e-3f;
  if (passed) {
    std::cerr << "p008 starter unexpectedly passed; TODO SwiGLU flow is incomplete\n";
    return 1;
  }
  std::cerr << "p008 starter intentionally fails validation until TODOs are completed\n";
  return 1;
}
