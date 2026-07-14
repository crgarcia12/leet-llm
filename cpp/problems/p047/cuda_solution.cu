#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
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

__global__ void fused_qkv_rope_kernel(const float* residual, const float* weights,
                                      float* qkv, int dim, int position) {
  const int proj = blockIdx.x;
  const int out = threadIdx.x;
  if (proj >= 3 || out >= dim) return;
  float sum = 0.0f;
  for (int in = 0; in < dim; ++in) sum += weights[(proj * dim + out) * dim + in] * residual[in];
  qkv[proj * dim + out] = sum;
  __syncthreads();
  if (proj < 2 && (out % 2) == 0 && out + 1 < dim) {
    const float angle = position * powf(10000.0f, -static_cast<float>(out) / dim);
    const float x = qkv[proj * dim + out];
    const float y = qkv[proj * dim + out + 1];
    qkv[proj * dim + out] = x * cosf(angle) - y * sinf(angle);
    qkv[proj * dim + out + 1] = x * sinf(angle) + y * cosf(angle);
  }
}

std::vector<int> tokenize(const std::string& prompt) {
  std::vector<int> out{1};
  for (char c : prompt) {
    if (c == 'a') out.push_back(2);
    else if (c == 'b') out.push_back(3);
    else if (c == 'c') out.push_back(4);
    else if (c == ' ') out.push_back(5);
    else if (c == '.') out.push_back(6);
    else throw std::invalid_argument("unsupported prompt byte");
  }
  return out;
}

std::vector<float> fused_cpu(const std::vector<float>& residual, const std::vector<float>& w,
                             int dim, int position) {
  std::vector<float> out(3 * dim, 0.0f);
  for (int p = 0; p < 3; ++p)
    for (int o = 0; o < dim; ++o)
      for (int i = 0; i < dim; ++i)
        out[p * dim + o] += w[(p * dim + o) * dim + i] * residual[i];
  for (int p = 0; p < 2; ++p) {
    for (int i = 0; i < dim; i += 2) {
      const float angle = position * std::pow(10000.0f, -static_cast<float>(i) / dim);
      const float x = out[p * dim + i], y = out[p * dim + i + 1];
      out[p * dim + i] = x * std::cos(angle) - y * std::sin(angle);
      out[p * dim + i + 1] = x * std::sin(angle) + y * std::cos(angle);
    }
  }
  return out;
}

int sample_token(const std::vector<float>& logits, SplitMix64& rng) {
  std::vector<int> idx(logits.size());
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) idx[i] = i;
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
  });
  idx.resize(5);
  float maxv = logits[idx[0]];
  std::vector<float> probs;
  probs.reserve(idx.size());
  float sum = 0.0f;
  for (int id : idx) {
    float p = std::exp((logits[id] - maxv) / 0.8f);
    probs.push_back(p);
    sum += p;
  }
  for (float& p : probs) p /= sum;
  float cumulative = 0.0f;
  const float draw = static_cast<float>(rng.unit());
  for (int i = 0; i < static_cast<int>(idx.size()); ++i) {
    cumulative += probs[i];
    if (draw < cumulative) return idx[i];
  }
  return idx.back();
}

int main() {
  constexpr int vocab = 7, dim = 4;
  const std::string prompt = "ab c.";
  const auto prompt_tokens = tokenize(prompt);
  if (prompt_tokens != std::vector<int>({1, 2, 3, 5, 4, 6})) return 1;

  std::vector<float> embedding(vocab * dim), weights(3 * dim * dim), output(vocab * dim);
  for (int i = 0; i < static_cast<int>(embedding.size()); ++i) embedding[i] = ((i * 5) % 17 - 8) / 7.0f;
  for (int i = 0; i < static_cast<int>(weights.size()); ++i) weights[i] = ((i * 3) % 11 - 5) / 9.0f;
  for (int i = 0; i < static_cast<int>(output.size()); ++i) output[i] = ((i * 7) % 13 - 6) / 8.0f;

  float *d_residual = nullptr, *d_weights = nullptr, *d_qkv = nullptr;
  CUDA_CHECK(cudaMalloc(&d_residual, dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_weights, weights.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_qkv, 3 * dim * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_weights, weights.data(), weights.size() * sizeof(float), cudaMemcpyHostToDevice));

  std::vector<int> generated;
  std::vector<std::vector<float>> cache_k;
  SplitMix64 rng(47);

  std::vector<int> context = prompt_tokens;
  const int max_new = 4;
  for (int step = 0; step < max_new; ++step) {
    const int token = context.back();
    std::vector<float> residual(dim);
    for (int i = 0; i < dim; ++i) residual[i] = embedding[token * dim + i];
    CUDA_CHECK(cudaMemcpy(d_residual, residual.data(), dim * sizeof(float), cudaMemcpyHostToDevice));
    fused_qkv_rope_kernel<<<3, dim>>>(d_residual, d_weights, d_qkv, dim, static_cast<int>(context.size() - 1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> gpu_qkv(3 * dim), cpu_qkv = fused_cpu(residual, weights, dim, static_cast<int>(context.size() - 1));
    CUDA_CHECK(cudaMemcpy(gpu_qkv.data(), d_qkv, gpu_qkv.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < static_cast<int>(gpu_qkv.size()); ++i)
      if (std::abs(gpu_qkv[i] - cpu_qkv[i]) > 1e-5f + 1e-4f * std::abs(cpu_qkv[i])) return 1;

    cache_k.push_back(std::vector<float>(gpu_qkv.begin() + dim, gpu_qkv.begin() + 2 * dim));

    std::vector<float> logits(vocab, 0.0f);
    for (int v = 0; v < vocab; ++v) {
      for (int i = 0; i < dim; ++i) logits[v] += output[v * dim + i] * gpu_qkv[i];
      logits[v] += 0.01f * static_cast<float>(cache_k.size());
    }
    const int next = sample_token(logits, rng);
    generated.push_back(next);
    if (next == 0) break;
    context.push_back(next);
  }

  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_weights));
  CUDA_CHECK(cudaFree(d_qkv));

  SplitMix64 verify_rng(47);
  std::vector<int> verify_context = prompt_tokens;
  std::vector<int> verify_generated;
  int verify_cache = 0;
  for (int step = 0; step < static_cast<int>(generated.size()); ++step) {
    std::vector<float> residual(dim);
    for (int i = 0; i < dim; ++i) residual[i] = embedding[verify_context.back() * dim + i];
    auto qkv = fused_cpu(residual, weights, dim, static_cast<int>(verify_context.size() - 1));
    ++verify_cache;
    std::vector<float> logits(vocab, 0.0f);
    for (int v = 0; v < vocab; ++v) {
      for (int i = 0; i < dim; ++i) logits[v] += output[v * dim + i] * qkv[i];
      logits[v] += 0.01f * static_cast<float>(verify_cache);
    }
    const int next = sample_token(logits, verify_rng);
    verify_generated.push_back(next);
    if (next == 0) break;
    verify_context.push_back(next);
  }

  if (generated != verify_generated) return 1;
  std::cout << "p047 capstone slice validated: tokenizer, fused QKV+RoPE, serial decode integration\n";
  return 0;
}
