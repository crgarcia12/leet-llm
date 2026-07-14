#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

__global__ void workload_kernel(float* buffer, int count, int inner_iterations) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  float value = buffer[idx];
  for (int i = 0; i < inner_iterations; ++i) value = value * 1.00001f + 0.0001f;
  buffer[idx] = value;
}

struct Stats {
  double median_ns;
  std::uint64_t p95_ns;
};

Stats summarize(std::vector<std::uint64_t> samples) {
  std::sort(samples.begin(), samples.end());
  const std::size_t n = samples.size();
  const double median = n % 2 == 0 ? (samples[n / 2 - 1] + samples[n / 2]) * 0.5 : samples[n / 2];
  const std::size_t rank = std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(0.95 * n)));
  return {median, samples[rank - 1]};
}

std::vector<std::uint64_t> profile_stage(int warmups, int trials, int launches_per_trial,
                                         int count, int inner_iterations) {
  float* d_buffer = nullptr;
  CUDA_CHECK(cudaMalloc(&d_buffer, count * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_buffer, 0, count * sizeof(float)));

  for (int i = 0; i < warmups; ++i) {
    workload_kernel<<<(count + 127) / 128, 128>>>(d_buffer, count, inner_iterations);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<std::uint64_t> samples;
  samples.reserve(static_cast<std::size_t>(trials * launches_per_trial));
  for (int trial = 0; trial < trials; ++trial) {
    for (int step = 0; step < launches_per_trial; ++step) {
      cudaEvent_t start, stop;
      CUDA_CHECK(cudaEventCreate(&start));
      CUDA_CHECK(cudaEventCreate(&stop));
      CUDA_CHECK(cudaEventRecord(start));
      workload_kernel<<<(count + 127) / 128, 128>>>(d_buffer, count, inner_iterations);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float elapsed_ms = 0.0f;
      CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
      CUDA_CHECK(cudaEventDestroy(start));
      CUDA_CHECK(cudaEventDestroy(stop));
      samples.push_back(static_cast<std::uint64_t>(elapsed_ms * 1'000'000.0f));
    }
  }
  CUDA_CHECK(cudaFree(d_buffer));
  return samples;
}

int main() {
  const int warmups = 1;
  const int trials = 3;
  const int decode_steps = 2;
  auto prefill_samples = profile_stage(warmups, trials, 1, 1 << 16, 256);
  auto decode_samples_ctx4 = profile_stage(warmups, trials, decode_steps, 1 << 12, 128);
  auto decode_samples_ctx16 = profile_stage(warmups, trials, decode_steps, 1 << 13, 128);

  const Stats prefill = summarize(prefill_samples);
  const Stats ctx4 = summarize(decode_samples_ctx4);
  const Stats ctx16 = summarize(decode_samples_ctx16);

  if (prefill_samples.size() != static_cast<std::size_t>(trials) ||
      decode_samples_ctx4.size() != static_cast<std::size_t>(trials * decode_steps) ||
      decode_samples_ctx16.size() != static_cast<std::size_t>(trials * decode_steps)) {
    return 1;
  }
  if (!(prefill.median_ns > 0 && ctx4.median_ns > 0 && ctx16.median_ns > 0 &&
        prefill.p95_ns >= static_cast<std::uint64_t>(prefill.median_ns * 0.5))) {
    return 1;
  }
  std::cout << "p044 prefill/decode profiling statistics produced deterministic sample counts\n";
  return 0;
}
