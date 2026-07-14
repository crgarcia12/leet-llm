#include "cuda_check.hpp"

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
  pair_ranks[i] = -1;
  pair_results[i] = -1;
  // TODO: scan merge rules and write the best (lowest-rank) merge for this adjacent pair.
}

std::vector<int> encode_starter(const std::string& text, const std::vector<MergeRule>&) {
  std::vector<int> tokens;
  tokens.reserve(text.size());
  for (unsigned char b : text) tokens.push_back(static_cast<int>(b));
  // TODO: repeatedly launch rank_pairs_kernel, choose the leftmost best rank, and merge until fixed point.
  return tokens;
}

int main() {
  const std::vector<MergeRule> rules = {
      {116, 104, 258, 0}, {258, 101, 259, 1}, {32, 259, 260, 2}, {195, 169, 261, 3}};
  const auto encoded = encode_starter("the the", rules);
  const std::vector<int> expected{259, 260};
  if (encoded == expected) {
    std::cerr << "Starter unexpectedly passed; complete TODOs with full ranked BPE merging.\n";
    return 1;
  }
  std::cout << "p037 starter builds. TODO: ranked merge loop is intentionally incomplete.\n";
  return 0;
}
