#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

struct Ranked {
  int token;
  float logit;
};

struct SamplerConfig {
  float temperature;
  int top_k;
  float top_p;
};

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

__global__ void stable_exp_kernel(const float* logits, float maximum, float temperature,
                                  float* exponentials, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) exponentials[i] = expf(logits[i] / temperature - maximum);
}

__global__ void normalize_kernel(const float* exponentials, float denominator, float* probabilities,
                                 int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) probabilities[i] = exponentials[i] / denominator;
}

std::vector<float> gpu_softmax(const std::vector<float>& logits, float temperature) {
  float maximum = -INFINITY;
  for (float v : logits) maximum = std::max(maximum, v / temperature);

  float *d_logits = nullptr, *d_exp = nullptr, *d_prob = nullptr;
  CUDA_CHECK(cudaMalloc(&d_logits, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_exp, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_prob, logits.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));

  stable_exp_kernel<<<(static_cast<int>(logits.size()) + 127) / 128, 128>>>(
      d_logits, maximum, temperature, d_exp, static_cast<int>(logits.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> exponentials(logits.size());
  CUDA_CHECK(cudaMemcpy(exponentials.data(), d_exp, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
  float denominator = std::accumulate(exponentials.begin(), exponentials.end(), 0.0f);

  normalize_kernel<<<(static_cast<int>(logits.size()) + 127) / 128, 128>>>(
      d_exp, denominator, d_prob, static_cast<int>(logits.size()));
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> probabilities(logits.size());
  CUDA_CHECK(cudaMemcpy(probabilities.data(), d_prob, logits.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_logits));
  CUDA_CHECK(cudaFree(d_exp));
  CUDA_CHECK(cudaFree(d_prob));
  return probabilities;
}

int sample_with_impl(const std::vector<float>& logits, const SamplerConfig& cfg, SplitMix64& rng,
                     bool use_gpu_softmax) {
  std::vector<Ranked> ranked;
  ranked.reserve(logits.size());
  for (int i = 0; i < static_cast<int>(logits.size()); ++i) ranked.push_back({i, logits[i]});
  std::sort(ranked.begin(), ranked.end(), [](const Ranked& a, const Ranked& b) {
    return a.logit == b.logit ? a.token < b.token : a.logit > b.logit;
  });

  ranked.resize(cfg.top_k > 0 ? std::min(cfg.top_k, static_cast<int>(ranked.size())) : ranked.size());
  std::vector<float> retained_logits;
  retained_logits.reserve(ranked.size());
  for (const auto& item : ranked) retained_logits.push_back(item.logit);

  std::vector<float> probabilities = use_gpu_softmax
      ? gpu_softmax(retained_logits, cfg.temperature)
      : [&] {
          float maxv = -INFINITY;
          for (float v : retained_logits) maxv = std::max(maxv, v / cfg.temperature);
          std::vector<float> ex(retained_logits.size());
          for (int i = 0; i < static_cast<int>(retained_logits.size()); ++i)
            ex[i] = std::exp(retained_logits[i] / cfg.temperature - maxv);
          float sum = std::accumulate(ex.begin(), ex.end(), 0.0f);
          for (float& v : ex) v /= sum;
          return ex;
        }();

  if (cfg.top_p > 0 && cfg.top_p < 1) {
    float cumulative = 0;
    int keep = 0;
    do {
      cumulative += probabilities[keep];
      ++keep;
    } while (keep < static_cast<int>(probabilities.size()) && cumulative < cfg.top_p);
    ranked.resize(keep);
    probabilities.resize(keep);
    const float sum = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    for (float& p : probabilities) p /= sum;
  }

  const double draw = rng.unit();
  double cumulative = 0;
  int selected = ranked.back().token;
  for (int i = 0; i < static_cast<int>(ranked.size()); ++i) {
    cumulative += probabilities[i];
    if (draw < cumulative) {
      selected = ranked[i].token;
      break;
    }
  }
  return selected;
}

int main() {
  const std::vector<float> tie_logits{1.0f, 1.0f, 0.999999f};
  const auto greedy = std::max_element(tie_logits.begin(), tie_logits.end()) - tie_logits.begin();
  if (greedy != 0) return 1;

  const std::vector<float> logits{4.0f, 3.0f, 2.0f, 1.0f};
  const SamplerConfig cfg{1.0f, 3, 0.8f};
  SplitMix64 gpu_rng(13), cpu_rng(13);
  const int gpu_selected = sample_with_impl(logits, cfg, gpu_rng, true);
  const int cpu_selected = sample_with_impl(logits, cfg, cpu_rng, false);
  if (gpu_selected != cpu_selected) return 1;

  const auto gpu_probs = gpu_softmax({10000.0f, 0.0f, -10000.0f}, 0.5f);
  if (!(gpu_probs[0] > 0.9999f && std::isfinite(gpu_probs[0]) && std::isfinite(gpu_probs[1]))) return 1;

  std::cout << "p038 logits filtering/sampling matches independent CPU reference\n";
  return 0;
}
