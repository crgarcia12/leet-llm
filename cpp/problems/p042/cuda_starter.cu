#include "cuda_check.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  // TODO: compare every named capture and report the first divergent one.
  // Current starter only checks final logits and misses earlier convention faults.
  std::vector<float> reference_logits{1.0f, 0.5f, -0.25f};
  std::vector<float> candidate_logits{1.001f, 0.5f, -0.25f};
  const bool final_close = std::abs(reference_logits[0] - candidate_logits[0]) < 1e-2f;
  std::cout << "p042 starter builds. TODO: first-divergence scan incomplete (logits check="
            << final_close << ")\n";
  return 0;
}
