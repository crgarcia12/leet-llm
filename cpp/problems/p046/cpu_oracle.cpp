#include "leetllm.hpp"
#include <iostream>
int main() {
  std::string detail;
  if (!leetllm::run_oracle(46, detail)) { std::cerr << detail << '\n'; return 1; }
  std::cout << "p046 CPU oracle passed: " << detail << '\n';
}
