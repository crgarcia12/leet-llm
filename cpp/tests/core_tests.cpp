#include "leetllm.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

int main() {
  for (int lesson = 1; lesson <= 47; ++lesson) {
    std::string detail;
    if (!leetllm::run_oracle(lesson, detail)) {
      std::cerr << detail << '\n';
      return lesson;
    }
  }
  const auto zero_rows = leetllm::stable_softmax({}, 0, 257);
  if (zero_rows.rows != 0 || zero_rows.columns != 257 || !zero_rows.values.empty()) return 48;
  for (const auto bad : {0, 1, 2}) {
    bool rejected = false;
    try {
      if (bad == 0) (void)leetllm::stable_softmax({}, 1, 0);
      if (bad == 1) (void)leetllm::stable_softmax({1}, 1, 1, 1);
      if (bad == 2) (void)leetllm::stable_softmax({std::numeric_limits<float>::quiet_NaN()}, 1, 1);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    if (!rejected) return 49 + bad;
  }
  std::cout << "47 CPU lesson oracles and softmax edge contracts passed\n";
}
