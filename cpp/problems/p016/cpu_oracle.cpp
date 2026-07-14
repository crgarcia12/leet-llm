#include "leetllm.hpp"
#include <iostream>
int main() {
  std::string detail;
  if (!leetllm::run_oracle(16, detail)) { std::cerr << detail << '\n'; return 1; }
  std::cout << "p016 CPU oracle passed: " << detail << '\n';
}
