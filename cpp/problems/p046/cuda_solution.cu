#include "cuda_check.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

struct SplitMix64 {
  std::uint64_t state;
  explicit SplitMix64(std::uint64_t seed) : state(seed) {}
  std::uint64_t next() {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  double unit() { return static_cast<double>(next() >> 11) / 9007199254740992.0; }
};

__global__ void correction_kernel(const float* target, const float* draft, float* correction, int vocab) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < vocab) correction[i] = fmaxf(target[i] - draft[i], 0.0f);
}

int sample_distribution(const std::vector<float>& probs, double draw) {
  double cumulative = 0.0;
  for (int i = 0; i < static_cast<int>(probs.size()); ++i) {
    cumulative += probs[i];
    if (draw < cumulative) return i;
  }
  return static_cast<int>(probs.size() - 1);
}

int main() {
  // Case 1: accept-all and bonus.
  {
    SplitMix64 rng(46);
    std::vector<int> emitted;
    for (int i = 0; i < 2; ++i) {
      const std::vector<float> draft{1.0f, 0.0f}, target{1.0f, 0.0f};
      const int token = sample_distribution(draft, rng.unit());
      const float ratio = std::min(1.0f, target[token] / draft[token]);
      if (rng.unit() <= ratio) emitted.push_back(token);
    }
    emitted.push_back(sample_distribution({0.0f, 1.0f}, rng.unit()));
    if (emitted != std::vector<int>({0, 0, 1})) return 1;
  }

  // Case 2: reject-first and sample correction distribution computed on CUDA.
  {
    SplitMix64 rng(460);
    const std::vector<float> draft{1.0f, 0.0f}, target{0.0f, 1.0f};
    const int proposed = 0;
    const float ratio = std::min(1.0f, target[proposed] / draft[proposed]);
    if (ratio != 0.0f) return 1;

    float *d_target = nullptr, *d_draft = nullptr, *d_corr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_target, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_draft, 2 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_corr, 2 * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_target, target.data(), 2 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_draft, draft.data(), 2 * sizeof(float), cudaMemcpyHostToDevice));
    correction_kernel<<<1, 32>>>(d_target, d_draft, d_corr, 2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> correction(2);
    CUDA_CHECK(cudaMemcpy(correction.data(), d_corr, 2 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_draft));
    CUDA_CHECK(cudaFree(d_corr));

    const float sum = correction[0] + correction[1];
    correction[0] /= sum;
    correction[1] /= sum;
    const int replacement = sample_distribution(correction, rng.unit());
    if (replacement != 1 || correction != std::vector<float>({0.0f, 1.0f})) return 1;
  }

  std::cout << "p046 speculative decoding accept/reject paths validated\n";
  return 0;
}
