#include "leetllm.hpp"

#include <charconv>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

#if LEETLLM_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

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
               "benchmark NNN --cuda [--iterations N] | "
               "report roofline --output FILE | capstone --backend cpu --prompt TEXT "
               "--max-tokens N --seed N\n";
}

std::filesystem::path cuda_solution_path(const char* executable, int lesson) {
  std::ostringstream name;
  name << 'p';
  if (lesson < 10) name << "00";
  else if (lesson < 100) name << '0';
  name << lesson << "_cuda_solution";
#ifdef _WIN32
  name << ".exe";
#endif
  auto directory = std::filesystem::absolute(executable).parent_path();
  return directory / name.str();
}

std::string lesson_id(int lesson) {
  std::ostringstream id;
  if (lesson < 10) id << "00";
  else if (lesson < 100) id << '0';
  id << lesson;
  return id.str();
}

void run_cuda_solution(const char* executable, int lesson) {
#if LEETLLM_ENABLE_CUDA
  const auto solution = cuda_solution_path(executable, lesson);
  if (!std::filesystem::exists(solution))
    throw std::runtime_error("CUDA solution executable not found: " + solution.string());
  const std::string command = '"' + solution.string() + '"';
  if (std::system(command.c_str()) != 0)
    throw std::runtime_error("CUDA solution failed for lesson " + std::to_string(lesson));
#else
  (void)executable;
  (void)lesson;
  throw std::runtime_error(
      "CUDA checks require a build configured with LEETLLM_ENABLE_CUDA=ON");
#endif
}

#if LEETLLM_ENABLE_CUDA
void require_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess)
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

std::string cuda_version(int value) {
  return std::to_string(value / 1000) + "." + std::to_string((value % 1000) / 10);
}
#endif
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
      const auto id = lesson_id(lesson);
      std::cout << "Lesson " << argv[2] << ": " << leetllm::lesson_titles()[lesson] << "\n"
                 << "Windows CPU: cmake --preset windows-cpu-debug; cmake --build --preset windows-cpu-debug\n"
                 << "NVIDIA CUDA stage: edit cpp/problems/p" << id
                 << "/cuda_starter.cu, then use preset windows-cuda-debug\n"
                 << "Canonical CUDA source: cpp/problems/p" << id << "/cuda_solution.cu\n";
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
      } else {
        run_cuda_solution(argv[0], lesson);
        if (lesson == 47) {
          std::cout << "fused QKV + RoPE CUDA output matches the CPU oracle "
                      "(absolute tolerance 1e-5, relative tolerance 1e-4)\n";
        } else {
          std::cout << "CUDA canonical result compared against the CPU oracle; "
                        "absolute tolerance 1e-5, relative tolerance 1e-4\n";
        }
      }
      return 0;
    }
    if (command == "benchmark") {
      if (argc != 4 && argc != 6)
        throw std::invalid_argument(
            "benchmark syntax is: benchmark NNN --cuda [--iterations N]");
      const int lesson = integer(argv[2], "lesson number");
      if (lesson < 1 || lesson > 47)
        throw std::out_of_range("benchmark lesson must be 001 through 047");
      if (std::string(argv[3]) != "--cuda")
        throw std::invalid_argument("benchmark backend must be --cuda");
      int iterations = 10;
      if (argc == 6) {
        if (std::string(argv[4]) != "--iterations")
          throw std::invalid_argument("expected --iterations");
        iterations = integer(argv[5], "iterations");
        if (iterations < 1) throw std::invalid_argument("iterations must be positive");
      }
      run_cuda_solution(argv[0], lesson);
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < iterations; ++i) run_cuda_solution(argv[0], lesson);
      const auto elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      std::cout << "lesson " << argv[2] << " CUDA benchmark: " << iterations
                << " measured end-to-end process iterations, average "
                << elapsed / iterations
                << " ms (includes process startup, transfers, and synchronization)\n";
      return 0;
    }
    if (command == "report") {
      if (argc != 5 || std::string(argv[2]) != "roofline" || std::string(argv[3]) != "--output")
        throw std::invalid_argument("report syntax is: report roofline --output FILE");
#if LEETLLM_ENABLE_CUDA
      constexpr int warmups = 1, iterations = 10;
      for (int i = 0; i < warmups; ++i) run_cuda_solution(argv[0], 6);
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < iterations; ++i) run_cuda_solution(argv[0], 6);
      const double elapsed = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count();
      cudaDeviceProp properties{};
      int driver_version = 0, runtime_version = 0;
      require_cuda(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
      require_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");
      require_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
      leetllm::write_roofline_report(
          argv[4], properties.name, cuda_version(driver_version),
          cuda_version(runtime_version), warmups, iterations, elapsed);
#else
      throw std::runtime_error(
          "roofline reporting requires a CUDA-enabled build and NVIDIA GPU");
#endif
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
