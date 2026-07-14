#include "cuda_check.hpp"

#include <iostream>
#include <vector>

int main() {
  // TODO: model buffer lifetimes, check overlap windows, and reuse aligned ranges.
  // Current starter intentionally uses naive non-reused layout.
  const std::vector<int> naive_offsets{0, 24, 32, 48};
  std::cout << "p041 starter builds. TODO: arena reuse planner incomplete (naive offsets: ";
  for (int v : naive_offsets) std::cout << v << ' ';
  std::cout << ")\n";
  return 0;
}
