#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace leetllm {

struct Tensor2D {
  std::size_t rows;
  std::size_t columns;
  std::vector<float> values;
};

const std::vector<std::string>& lesson_titles();
Tensor2D stable_softmax(const std::vector<float>& values, std::size_t rows,
                        std::size_t columns, std::size_t rank = 2);
bool run_oracle(int lesson, std::string& detail);
std::vector<int> capstone_generate(const std::string& prompt, int max_tokens,
                                   std::uint64_t seed);
void write_roofline_report(const std::filesystem::path& output);

}  // namespace leetllm

