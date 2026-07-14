#include "cuda_check.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

struct MergeRule {
  int left;
  int right;
  int result;
  int rank;
};

__global__ void rank_pairs_kernel(const int* tokens, int token_count,
                                  const MergeRule* rules, int rule_count,
                                  int* pair_ranks, int* pair_results) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= token_count - 1) return;
  int best_rank = 1 << 30;
  int best_result = -1;
  const int left = tokens[i];
  const int right = tokens[i + 1];
  for (int r = 0; r < rule_count; ++r) {
    if (rules[r].left == left && rules[r].right == right && rules[r].rank < best_rank) {
      best_rank = rules[r].rank;
      best_result = rules[r].result;
    }
  }
  pair_ranks[i] = best_result < 0 ? -1 : best_rank;
  pair_results[i] = best_result;
}

std::vector<int> encode_cpu(const std::string& text, const std::vector<MergeRule>& rules) {
  std::vector<int> tokens;
  tokens.reserve(text.size());
  for (unsigned char b : text) tokens.push_back(static_cast<int>(b));
  while (tokens.size() >= 2) {
    int best_index = -1;
    int best_rank = 1 << 30;
    int best_result = -1;
    for (int i = 0; i + 1 < static_cast<int>(tokens.size()); ++i) {
      for (const auto& rule : rules) {
        if (rule.left == tokens[i] && rule.right == tokens[i + 1] && rule.rank < best_rank) {
          best_rank = rule.rank;
          best_index = i;
          best_result = rule.result;
        }
      }
    }
    if (best_index < 0) break;
    tokens[best_index] = best_result;
    tokens.erase(tokens.begin() + best_index + 1);
  }
  return tokens;
}

std::vector<int> encode_gpu(const std::string& text, const std::vector<MergeRule>& rules) {
  std::vector<int> tokens;
  tokens.reserve(text.size());
  for (unsigned char b : text) tokens.push_back(static_cast<int>(b));

  MergeRule* d_rules = nullptr;
  CUDA_CHECK(cudaMalloc(&d_rules, rules.size() * sizeof(MergeRule)));
  CUDA_CHECK(cudaMemcpy(d_rules, rules.data(), rules.size() * sizeof(MergeRule), cudaMemcpyHostToDevice));

  while (tokens.size() >= 2) {
    const int pair_count = static_cast<int>(tokens.size()) - 1;
    int *d_tokens = nullptr, *d_ranks = nullptr, *d_results = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, tokens.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ranks, pair_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_results, pair_count * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_tokens, tokens.data(), tokens.size() * sizeof(int), cudaMemcpyHostToDevice));

    rank_pairs_kernel<<<(pair_count + 127) / 128, 128>>>(
        d_tokens, static_cast<int>(tokens.size()), d_rules, static_cast<int>(rules.size()), d_ranks, d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> ranks(pair_count), results(pair_count);
    CUDA_CHECK(cudaMemcpy(ranks.data(), d_ranks, pair_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(results.data(), d_results, pair_count * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_ranks));
    CUDA_CHECK(cudaFree(d_results));

    int best_index = -1;
    int best_rank = 1 << 30;
    int best_result = -1;
    for (int i = 0; i < pair_count; ++i) {
      if (ranks[i] >= 0 && ranks[i] < best_rank) {
        best_rank = ranks[i];
        best_index = i;
        best_result = results[i];
      }
    }
    if (best_index < 0) break;
    tokens[best_index] = best_result;
    tokens.erase(tokens.begin() + best_index + 1);
  }

  CUDA_CHECK(cudaFree(d_rules));
  return tokens;
}

int main() {
  const std::vector<MergeRule> rules = {
      {116, 104, 258, 0}, {258, 101, 259, 1}, {32, 259, 260, 2}, {195, 169, 261, 3},
      {108, 108, 262, 4}, {101, 262, 263, 5}, {104, 263, 264, 6}, {264, 111, 265, 7}};

  const std::string ascii = "the the thth";
  const std::string unicode = u8"café";
  const auto gpu_ascii = encode_gpu(ascii, rules);
  const auto gpu_unicode = encode_gpu(unicode, rules);
  const auto cpu_ascii = encode_cpu(ascii, rules);
  const auto cpu_unicode = encode_cpu(unicode, rules);

  const std::vector<int> expected_ascii{259, 260, 32, 258, 258};
  const std::vector<int> expected_unicode{99, 97, 102, 261};
  if (gpu_ascii != expected_ascii || gpu_unicode != expected_unicode ||
      gpu_ascii != cpu_ascii || gpu_unicode != cpu_unicode) {
    return 1;
  }
  std::cout << "p037 byte-BPE ranked merges validated against independent CPU reference\n";
  return 0;
}
