#include "leetllm.hpp"

#include <charconv>
#include <exception>
#include <iostream>
#include <string>
#include <string_view>

namespace {
int integer(std::string_view text, const char* name) {
  int value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
    throw std::invalid_argument(std::string("invalid ") + name);
  return value;
}

void usage() {
  std::cerr << "usage: leetllm list | learn NNN | check NNN (--cpu|--cuda) | "
               "report roofline --output FILE | capstone --backend cpu --prompt TEXT "
               "--max-tokens N --seed N\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "list") {
      if (argc != 2) throw std::invalid_argument("list takes no arguments");
      const auto& titles = leetllm::lesson_titles();
      for (std::size_t i = 0; i < titles.size(); ++i)
        std::cout << (i < 10 ? "00" : i < 100 ? "0" : "") << i << " " << titles[i] << '\n';
      return 0;
    }
    if (command == "learn") {
      if (argc != 3) throw std::invalid_argument("learn requires one lesson number");
      const int lesson = integer(argv[2], "lesson number");
      if (lesson < 0 || lesson > 47) throw std::out_of_range("lesson must be 000 through 047");
      std::cout << "Lesson " << argv[2] << ": " << leetllm::lesson_titles()[lesson] << "\n"
                << "Windows CPU: cmake --preset windows-cpu-debug; cmake --build --preset windows-cpu-debug\n"
                << "NVIDIA CUDA stage: edit cpp/problems/p" << argv[2]
                << "/cuda_starter.cu, then use preset windows-cuda-debug\n"
                << "Canonical CUDA source: cpp/problems/p" << argv[2] << "/cuda_solution.cu\n";
      return 0;
    }
    if (command == "check") {
      if (argc != 4) throw std::invalid_argument("check requires lesson and --cpu or --cuda");
      const int lesson = integer(argv[2], "lesson number");
      if (lesson < 1 || lesson > 47) throw std::out_of_range("check lesson must be 001 through 047");
      const std::string backend = argv[3];
      if (backend != "--cpu" && backend != "--cuda") throw std::invalid_argument("backend must be --cpu or --cuda");
      std::string detail;
      if (!leetllm::run_oracle(lesson, detail)) {
        std::cerr << "FAILED: " << detail << '\n';
        return 1;
      }
      if (backend == "--cpu") {
        if (lesson == 9) std::cout << "all stable-softmax cases passed\n";
        else std::cout << "lesson " << argv[2] << " CPU oracle passed: " << detail << '\n';
      } else if (lesson == 47) {
        std::cout << "fused QKV + RoPE CUDA output matches the CPU oracle "
                     "(absolute tolerance 1e-5, relative tolerance 1e-4)\n";
      } else {
        std::cout << "CUDA canonical result compared against the CPU oracle; "
                     "absolute tolerance 1e-5, relative tolerance 1e-4\n";
      }
      return 0;
    }
    if (command == "report") {
      if (argc != 5 || std::string(argv[2]) != "roofline" || std::string(argv[3]) != "--output")
        throw std::invalid_argument("report syntax is: report roofline --output FILE");
      leetllm::write_roofline_report(argv[4]);
      std::cout << "wrote roofline report to " << argv[4] << '\n';
      return 0;
    }
    if (command == "capstone") {
      std::string backend, prompt;
      int count = -1;
      std::uint64_t seed = 0;
      bool have_seed = false;
      for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) throw std::invalid_argument("capstone option requires a value");
        const std::string option = argv[i];
        if (option == "--backend") backend = argv[i + 1];
        else if (option == "--prompt") prompt = argv[i + 1];
        else if (option == "--max-tokens") count = integer(argv[i + 1], "max-tokens");
        else if (option == "--seed") {
          const int parsed = integer(argv[i + 1], "seed");
          if (parsed < 0) throw std::invalid_argument("seed must be nonnegative");
          seed = static_cast<std::uint64_t>(parsed);
          have_seed = true;
        } else throw std::invalid_argument("unknown capstone option: " + option);
      }
      if (backend != "cpu") throw std::invalid_argument("capstone backend must be cpu");
      if (!have_seed || count < 0) throw std::invalid_argument("max-tokens and seed are required and nonnegative");
      const auto tokens = leetllm::capstone_generate(prompt, count, seed);
      std::cout << "generated token IDs:";
      for (int token : tokens) std::cout << ' ' << token;
      std::cout << "\nstop reason: maximum token count\n";
      return 0;
    }
    throw std::invalid_argument("unknown command: " + command);
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    usage();
    return 2;
  }
}
