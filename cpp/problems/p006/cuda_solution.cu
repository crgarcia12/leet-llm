#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct RooflineWorkload {
  double flops;
  double bytes;
};

struct RooflineMachine {
  double peak_compute_gflops;
  double peak_bandwidth_gbps;
};

struct RooflinePrediction {
  double intensity;
  double bandwidth_ceiling;
  double predicted_ceiling;
  enum class Bottleneck { Memory, Compute, Balanced } bottleneck;
};

RooflinePrediction predict(const RooflineWorkload& workload, const RooflineMachine& machine) {
  const double intensity = workload.flops / workload.bytes;
  const double bw_ceiling = intensity * machine.peak_bandwidth_gbps;
  RooflinePrediction::Bottleneck bottleneck = RooflinePrediction::Bottleneck::Balanced;
  if (bw_ceiling < machine.peak_compute_gflops) bottleneck = RooflinePrediction::Bottleneck::Memory;
  if (bw_ceiling > machine.peak_compute_gflops) bottleneck = RooflinePrediction::Bottleneck::Compute;
  return RooflinePrediction{intensity, bw_ceiling, std::min(machine.peak_compute_gflops, bw_ceiling), bottleneck};
}

__global__ void streaming_triad_kernel(const float* x, float* y, float alpha, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) y[index] = alpha * x[index] + y[index];
}

void cpu_streaming_triad(const std::vector<float>& x, std::vector<float>& y, float alpha, int iterations) {
  for (int it = 0; it < iterations; ++it)
    for (std::size_t i = 0; i < x.size(); ++i)
      y[i] = alpha * x[i] + y[i];
}

bool validate_triad() {
  constexpr int count = 4096;
  constexpr int iterations = 7;
  constexpr float alpha = 1.25f;

  std::vector<float> x(count), y_cpu(count), y_gpu(count);
  for (int i = 0; i < count; ++i) {
    x[i] = static_cast<float>((i % 23) - 11) / 7.0f;
    y_cpu[i] = static_cast<float>((i % 13) - 6) / 5.0f;
  }
  y_gpu = y_cpu;

  cpu_streaming_triad(x, y_cpu, alpha, iterations);

  float *device_x = nullptr, *device_y = nullptr;
  CUDA_CHECK(cudaMalloc(&device_x, count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&device_y, count * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(device_x, x.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_y, y_gpu.data(), count * sizeof(float), cudaMemcpyHostToDevice));

  const int threads = 256;
  const int blocks = (count + threads - 1) / threads;
  for (int it = 0; it < iterations; ++it) {
    streaming_triad_kernel<<<blocks, threads>>>(device_x, device_y, alpha, count);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(y_gpu.data(), device_y, count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(device_x));
  CUDA_CHECK(cudaFree(device_y));

  for (int i = 0; i < count; ++i) {
    if (std::abs(y_cpu[i] - y_gpu[i]) > 1e-5f * std::max({1.0f, std::abs(y_cpu[i]), std::abs(y_gpu[i])})) {
      std::cerr << "triad mismatch at " << i << " expected " << y_cpu[i] << " got " << y_gpu[i] << '\n';
      return false;
    }
  }
  return true;
}

bool validate_model() {
  const RooflineMachine machine{1000.0, 100.0};
  const RooflinePrediction stream = predict({2000.0, 8000.0}, machine);
  const RooflinePrediction ridge = predict({1000.0, 100.0}, machine);

  const bool stream_ok = std::abs(stream.intensity - 0.25) < 1e-12 &&
                         std::abs(stream.bandwidth_ceiling - 25.0) < 1e-9 &&
                         std::abs(stream.predicted_ceiling - 25.0) < 1e-9 &&
                         stream.bottleneck == RooflinePrediction::Bottleneck::Memory;

  const bool ridge_ok = std::abs(ridge.intensity - 10.0) < 1e-12 &&
                        std::abs(ridge.predicted_ceiling - 1000.0) < 1e-9 &&
                        ridge.bottleneck == RooflinePrediction::Bottleneck::Balanced;

  if (!stream_ok || !ridge_ok) {
    std::cerr << "roofline model validation failed\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  if (!validate_model()) return 1;
  if (!validate_triad()) return 1;
  std::cout << "p006 CUDA canonical solution passed roofline-model and streaming-kernel validation\n";
  return 0;
}
