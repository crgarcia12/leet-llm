#include "leetllm.hpp"
#include <iostream>
int main() {
  std::string detail;
  if (!leetllm::run_oracle(23, detail)) { std::cerr << detail << '\n'; return 1; }
  std::cout << "p023 CPU oracle passed: " << detail << '\n';
}
