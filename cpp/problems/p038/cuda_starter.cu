#include "cuda_check.hpp"

#include <algorithm>
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
};

int sample_starter(const std::vector<float>& logits, SplitMix64& rng) {
  const auto best = std::max_element(logits.begin(), logits.end()) - logits.begin();
  (void)rng;
  // TODO: implement temperature/top-k/top-p filtering, stable softmax, and one seeded draw.
  return static_cast<int>(best);
}

int main() {
  SplitMix64 rng(13);
  const int selected = sample_starter({4, 3, 2, 1}, rng);
  if (selected == 0) {
    std::cout << "p038 starter builds. TODO: stochastic sampling path is intentionally incomplete.\n";
    return 0;
  }
  return 1;
}
