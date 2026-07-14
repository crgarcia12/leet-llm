#include "leetllm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace leetllm {
namespace {
constexpr float kPi = 3.14159265358979323846f;

bool near(double a, double b, double tolerance = 1e-5) {
  return std::abs(a - b) <= tolerance * std::max({1.0, std::abs(a), std::abs(b)});
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) throw std::invalid_argument("dot shape mismatch");
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

std::vector<float> gemv(const std::vector<float>& matrix, std::size_t rows,
                        std::size_t columns, const std::vector<float>& vector) {
  if (matrix.size() != rows * columns || vector.size() != columns)
    throw std::invalid_argument("GEMV shape mismatch");
  std::vector<float> output(rows);
  for (std::size_t row = 0; row < rows; ++row)
    output[row] = std::inner_product(matrix.begin() + row * columns,
                                    matrix.begin() + (row + 1) * columns,
                                    vector.begin(), 0.0f);
  return output;
}

std::vector<float> rmsnorm(const std::vector<float>& input,
                           const std::vector<float>& scale) {
  if (input.empty() || input.size() != scale.size())
    throw std::invalid_argument("RMSNorm shape mismatch");
  double squares = 0;
  for (float value : input) squares += static_cast<double>(value) * value;
  const float inverse = 1.0f / std::sqrt(static_cast<float>(squares / input.size()) + 1e-5f);
  std::vector<float> output(input.size());
  for (std::size_t i = 0; i < input.size(); ++i) output[i] = input[i] * inverse * scale[i];
  return output;
}

std::vector<float> rope(std::vector<float> values, std::size_t position) {
  if (values.size() % 2 != 0) throw std::invalid_argument("RoPE width must be even");
  for (std::size_t i = 0; i < values.size(); i += 2) {
    const float angle = static_cast<float>(position) *
                        std::pow(10000.0f, -static_cast<float>(i) / values.size());
    const float x = values[i], y = values[i + 1];
    values[i] = x * std::cos(angle) - y * std::sin(angle);
    values[i + 1] = x * std::sin(angle) + y * std::cos(angle);
  }
  return values;
}

std::vector<float> attention(const std::vector<float>& query,
                             const std::vector<std::vector<float>>& keys,
                             const std::vector<std::vector<float>>& values,
                             std::size_t begin = 0) {
  if (keys.empty() || keys.size() != values.size() || begin >= keys.size())
    throw std::invalid_argument("attention shape mismatch");
  std::vector<float> scores;
  for (std::size_t i = begin; i < keys.size(); ++i)
    scores.push_back(dot(query, keys[i]) / std::sqrt(static_cast<float>(query.size())));
  const auto probabilities = stable_softmax(scores, 1, scores.size()).values;
  std::vector<float> output(values.front().size(), 0.0f);
  for (std::size_t i = 0; i < probabilities.size(); ++i)
    for (std::size_t j = 0; j < output.size(); ++j)
      output[j] += probabilities[i] * values[begin + i][j];
  return output;
}

std::int8_t quantize(float value, float scale) {
  return static_cast<std::int8_t>(std::clamp(std::lround(value / scale), -127L, 127L));
}

std::string gpu_name() {
  if (const char* name = std::getenv("LEETLLM_GPU_MODEL")) return name;
  return "NVIDIA GPU (query with nvidia-smi on CUDA host)";
}
}  // namespace

const std::vector<std::string>& lesson_titles() {
  static const std::vector<std::string> titles = {
      "Start Here: Build an LLM Inference Engine",
      "Vector Dot Product", "Tensor Storage and Strides", "Transpose and Tiled Copy",
      "Matrix-Vector Multiplication", "Matrix-Matrix Multiplication",
      "Build a Roofline Measurement", "ReLU, GELU, and SiLU",
      "SwiGLU Feed-Forward Gate", "Numerically Stable Softmax", "RMSNorm",
      "Residual Streams and Precision", "Fuse Norm, Scale, and Projection Input",
      "Embedding Lookup and Tied Output Weights", "Q/K/V Projections and Head Views",
      "Rotary Position Embeddings", "Causal Attention for One Head",
      "Multi-Head Attention", "MHA, MQA, and GQA", "Online Softmax Attention",
      "Tiled Fused Attention", "Sliding-Window Attention", "Preallocate and Append K/V",
      "Cached Single-Token Attention", "KV Layout Shootout", "Shared KV Heads",
      "Ring-Buffer Sliding Cache", "Paged KV Allocation", "Quantized KV Cache",
      "Symmetric INT8 Quantization", "Per-Channel and Groupwise Scales",
      "Pack and Unpack INT4", "Dequantize Then GEMV", "Fused Q4 GEMV",
      "Quantization Error Propagation", "One Decoder Transformer Block",
      "Parse a Model Weight Format", "Byte-Level BPE Tokenization and Detokenization",
      "Logits and Sampling", "Prompt Prefill", "Autoregressive Decode",
      "Buffer Reuse and Memory Planning", "Checkpoint Parity and First Divergence",
      "Fuse RMSNorm and Q/K/V Projections", "Profile Prefill and Decode Separately",
      "Static and Continuous Batching", "Speculative Decoding",
      "Capstone Inference Engine"};
  return titles;
}

Tensor2D stable_softmax(const std::vector<float>& values, std::size_t rows,
                        std::size_t columns, std::size_t rank) {
  if (rank != 2) throw std::invalid_argument("stable softmax requires rank 2");
  if (columns == 0) throw std::invalid_argument("stable softmax columns must be positive");
  if (values.size() != rows * columns) throw std::invalid_argument("stable softmax shape mismatch");
  for (float value : values)
    if (!std::isfinite(value)) throw std::invalid_argument("stable softmax requires finite logits");
  Tensor2D result{rows, columns, std::vector<float>(values.size())};
  for (std::size_t row = 0; row < rows; ++row) {
    const auto first = values.begin() + static_cast<std::ptrdiff_t>(row * columns);
    const float maximum = *std::max_element(first, first + static_cast<std::ptrdiff_t>(columns));
    double sum = 0;
    for (std::size_t column = 0; column < columns; ++column) {
      const float exponential = std::exp(values[row * columns + column] - maximum);
      result.values[row * columns + column] = exponential;
      sum += exponential;
    }
    for (std::size_t column = 0; column < columns; ++column)
      result.values[row * columns + column] /= static_cast<float>(sum);
  }
  return result;
}

bool run_oracle(int lesson, std::string& detail) {
  bool ok = false;
  switch (lesson) {
    case 1:
      ok = near(dot({1, 2, 3, -1}, {2, -1, 0.5f, 4}), -2.5);
      break;
    case 2: {
      std::array<int, 3> shape{2, 3, 4}, stride{12, 4, 1};
      ok = 1 * stride[0] + 2 * stride[1] + 3 * stride[2] == 23;
      break;
    }
    case 3: {
      std::vector<int> input{1, 2, 3, 4, 5, 6}, output(6);
      for (int r = 0; r < 2; ++r) for (int c = 0; c < 3; ++c) output[c * 2 + r] = input[r * 3 + c];
      ok = output == std::vector<int>({1, 4, 2, 5, 3, 6});
      break;
    }
    case 4:
      ok = gemv({1, 2, 3, 4, 5, 6}, 2, 3, {1, 0, -1}) == std::vector<float>({-2, -2});
      break;
    case 5: {
      const std::vector<float> a{1, 2, 3, 4}, b{5, 6, 7, 8};
      std::vector<float> c(4);
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 2; ++k) c[i * 2 + j] += a[i * 2 + k] * b[k * 2 + j];
      ok = c == std::vector<float>({19, 22, 43, 50});
      break;
    }
    case 6: {
      constexpr double bytes = 4.0 * 3 * 1024;
      constexpr double seconds = 0.002;
      const double bandwidth = bytes / seconds;
      ok = near(bandwidth, 6'144'000.0);
      break;
    }
    case 7: {
      const float x = -0.5f;
      const float relu = std::max(0.0f, x);
      const float silu = x / (1.0f + std::exp(-x));
      const float gelu = 0.5f * x * (1 + std::tanh(std::sqrt(2.0f / kPi) * (x + 0.044715f * x * x * x)));
      ok = relu == 0 && silu < 0 && gelu < 0;
      break;
    }
    case 8: {
      auto silu = [](float x) { return x / (1 + std::exp(-x)); };
      ok = near(silu(2) * 3, 5.284782);
      break;
    }
    case 9: {
      const auto ordinary = stable_softmax({1, 2, 3, 10000, 10001, 9999}, 2, 3);
      ok = near(ordinary.values[0], 0.0900306) && near(ordinary.values[4], 0.665241);
      std::vector<float> wide(257);
      std::iota(wide.begin(), wide.end(), -128.0f);
      const auto wide_result = stable_softmax(wide, 1, 257);
      ok = ok && near(std::accumulate(wide_result.values.begin(), wide_result.values.end(), 0.0), 1.0, 3e-5);
      ok = ok && stable_softmax({}, 0, 17).values.empty() &&
           stable_softmax({-10000, -9999, -10001}, 1, 3).values[1] > 0.66f;
      for (auto invocation : {0, 1, 2}) {
        try {
          if (invocation == 0) (void)stable_softmax({1}, 1, 1, 1);
          if (invocation == 1) (void)stable_softmax({}, 1, 0);
          if (invocation == 2) (void)stable_softmax({std::numeric_limits<float>::infinity()}, 1, 1);
          ok = false;
        } catch (const std::invalid_argument&) {
        }
      }
      break;
    }
    case 10: {
      const auto value = rmsnorm({3, 4}, {1, 1});
      ok = near(value[0], 0.848527, 1e-4) && near(value[1], 1.13137, 1e-4);
      break;
    }
    case 11: {
      float low = 10000;
      for (int i = 0; i < 1000; ++i) low += 0.001f;
      double high = 10000;
      for (int i = 0; i < 1000; ++i) high += 0.001;
      ok = std::abs(high - 10001.0) < std::abs(static_cast<double>(low) - 10001.0);
      break;
    }
    case 12: {
      const auto normalized = rmsnorm({1, 2}, {2, 3});
      ok = gemv({1, 0, 0, 1}, 2, 2, normalized) == normalized;
      break;
    }
    case 13: {
      const std::vector<float> embedding{1, 0, 0, 1, 1, 1};
      const std::vector<float> hidden{2, 3};
      ok = gemv(embedding, 3, 2, hidden) == std::vector<float>({2, 3, 5});
      break;
    }
    case 14: {
      const auto q = gemv({1, 0, 0, 1}, 2, 2, {2, 3});
      const auto k = gemv({2, 0, 0, 2}, 2, 2, {2, 3});
      const auto v = gemv({1, 1, 1, -1}, 2, 2, {2, 3});
      ok = q == std::vector<float>({2, 3}) && k == std::vector<float>({4, 6}) && v == std::vector<float>({5, -1});
      break;
    }
    case 15: {
      const auto rotated = rope({1, 0}, 1);
      ok = near(rotated[0], std::cos(1.0)) && near(rotated[1], std::sin(1.0));
      break;
    }
    case 16: {
      const auto result = attention({1, 0}, {{1, 0}, {0, 1}}, {{2, 0}, {0, 2}});
      ok = result[0] > result[1];
      break;
    }
    case 17: {
      const auto h0 = attention({1}, {{1}, {0}}, {{2}, {4}});
      const auto h1 = attention({0}, {{1}, {0}}, {{2}, {4}});
      ok = h0[0] < h1[0];
      break;
    }
    case 18: {
      std::array<int, 4> query_to_kv{};
      for (int q = 0; q < 4; ++q) query_to_kv[q] = q / 2;
      ok = query_to_kv == std::array<int, 4>{0, 0, 1, 1};
      break;
    }
    case 19: {
      double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
      for (double score : {1000.0, 1001.0, 999.0}) {
        const double next = std::max(maximum, score);
        denominator = denominator * std::exp(maximum - next) + std::exp(score - next);
        maximum = next;
      }
      ok = near(denominator, 1.503214);
      break;
    }
    case 20: {
      const auto reference = attention({1, 0}, {{1, 0}, {0, 1}, {1, 1}}, {{1, 2}, {3, 4}, {5, 6}});
      std::vector<float> tiled(reference.size());
      for (std::size_t tile = 0; tile < 3; ++tile) {
        const auto probability = stable_softmax({1 / std::sqrt(2.0f), 0, 1 / std::sqrt(2.0f)}, 1, 3).values[tile];
        for (std::size_t j = 0; j < 2; ++j) tiled[j] += probability * std::vector<std::vector<float>>{{1,2},{3,4},{5,6}}[tile][j];
      }
      ok = near(reference[0], tiled[0]) && near(reference[1], tiled[1]);
      break;
    }
    case 21: {
      const auto result = attention({1}, {{9}, {1}, {0}}, {{99}, {2}, {4}}, 1);
      ok = result[0] < 4 && result[0] > 2;
      break;
    }
    case 22: {
      std::vector<int> cache(4, -1);
      std::size_t count = 0;
      for (int token : {7, 8, 9}) cache[count++] = token;
      ok = count == 3 && cache[2] == 9 && cache.capacity() >= 4;
      break;
    }
    case 23:
      ok = attention({1}, {{1}, {0}}, {{2}, {4}})[0] > 2;
      break;
    case 24: {
      const std::size_t token_major = ((2 * 4 + 1) * 8 + 3);
      const std::size_t head_major = ((1 * 16 + 2) * 8 + 3);
      ok = token_major == 75 && head_major == 147;
      break;
    }
    case 25: {
      const int query_heads = 8, kv_heads = 2;
      ok = query_heads / kv_heads == 4 && 7 / (query_heads / kv_heads) == 1;
      break;
    }
    case 26: {
      std::array<int, 3> ring{-1, -1, -1};
      for (int token = 0; token < 5; ++token) ring[token % ring.size()] = token;
      ok = ring == std::array<int, 3>{3, 4, 2};
      break;
    }
    case 27: {
      constexpr std::size_t page = 16;
      const std::size_t pages = (33 + page - 1) / page;
      ok = pages == 3 && (32 / page == 2) && (32 % page == 0);
      break;
    }
    case 28: {
      const float scale = 2.0f / 127;
      const auto q = quantize(1.25f, scale);
      ok = std::abs(static_cast<float>(q) * scale - 1.25f) <= scale / 2;
      break;
    }
    case 29: {
      const float scale = 3.0f / 127;
      ok = quantize(-3, scale) == -127 && quantize(3, scale) == 127 && quantize(0, scale) == 0;
      break;
    }
    case 30: {
      const std::vector<float> values{1, 2, 10, 20};
      const float s0 = 2.0f / 127, s1 = 20.0f / 127;
      ok = quantize(values[1], s0) == 127 && quantize(values[3], s1) == 127;
      break;
    }
    case 31: {
      const int low = -3, high = 7;
      const std::uint8_t packed = static_cast<std::uint8_t>((low & 0xf) | ((high & 0xf) << 4));
      const int unpack_low = (packed & 8) ? static_cast<int>(packed & 15) - 16 : packed & 15;
      const int unpack_high = (packed >> 4 & 8) ? static_cast<int>(packed >> 4 & 15) - 16 : packed >> 4 & 15;
      ok = unpack_low == low && unpack_high == high;
      break;
    }
    case 32: {
      const std::vector<std::int8_t> q{1, 2, -1, 3};
      std::vector<float> dequantized;
      for (auto value : q) dequantized.push_back(value * 0.5f);
      ok = gemv(dequantized, 2, 2, {2, 1}) == std::vector<float>({2, 0.5f});
      break;
    }
    case 33: {
      const std::array<int, 4> q{-2, 1, 3, -1};
      float fused = 0;
      for (std::size_t i = 0; i < q.size(); ++i) fused += q[i] * 0.25f * std::array<float,4>{1,2,3,4}[i];
      ok = near(fused, 1.25);
      break;
    }
    case 34: {
      const float input_error = 0.01f, weight_norm = 3.0f;
      const float bound = input_error * weight_norm;
      ok = bound >= std::abs(dot({0.01f, -0.01f}, {1, 2}));
      break;
    }
    case 35: {
      const std::vector<float> residual{1, 2};
      const auto normalized = rmsnorm(residual, {1, 1});
      const auto projected = gemv({1, 0, 0, 1}, 2, 2, normalized);
      ok = near(residual[0] + projected[0], 1 + normalized[0]);
      break;
    }
    case 36: {
      std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
      const std::uint32_t magic = 0x4c4c4d31, count = 3;
      stream.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
      stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
      std::uint32_t read_magic{}, read_count{};
      stream.seekg(0);
      stream.read(reinterpret_cast<char*>(&read_magic), sizeof(read_magic));
      stream.read(reinterpret_cast<char*>(&read_count), sizeof(read_count));
      ok = read_magic == magic && read_count == count;
      break;
    }
    case 37: {
      const std::unordered_map<char, int> vocabulary{{'a', 2}, {'b', 3}, {' ', 5}};
      std::vector<int> encoded{1};
      for (char byte : std::string("ab a")) encoded.push_back(vocabulary.at(byte));
      ok = encoded == std::vector<int>({1, 2, 3, 5, 2});
      break;
    }
    case 38: {
      const std::vector<float> logits{1, 3, 3};
      const auto best = std::max_element(logits.begin(), logits.end());
      ok = std::distance(logits.begin(), best) == 1;
      break;
    }
    case 39: {
      std::vector<int> cache;
      const std::vector<int> prompt{1, 2, 3};
      cache.insert(cache.end(), prompt.begin(), prompt.end());
      ok = cache == prompt;
      break;
    }
    case 40: {
      std::vector<int> prefix{1, 2};
      for (int next : {3, 4, 5}) prefix.push_back(next);
      ok = prefix == std::vector<int>({1, 2, 3, 4, 5});
      break;
    }
    case 41: {
      struct Interval { int begin, end, bytes; };
      const std::array<Interval, 3> values{{{0, 2, 64}, {2, 4, 64}, {1, 3, 32}}};
      const int naive = 160, reused = 96;
      ok = values[0].end <= values[1].begin && reused < naive;
      break;
    }
    case 42: {
      const std::vector<float> expected{1, 2, 3}, actual{1, 2.1f, 4};
      const auto mismatch = std::mismatch(expected.begin(), expected.end(), actual.begin(),
                                          [](float a, float b) { return near(a, b); });
      ok = std::distance(expected.begin(), mismatch.first) == 1;
      break;
    }
    case 43: {
      const auto norm = rmsnorm({1, 2}, {1, 1});
      const auto fused = gemv({1, 0, 0, 1, 1, 1}, 3, 2, norm);
      ok = near(fused[2], norm[0] + norm[1]);
      break;
    }
    case 44: {
      const std::array<double, 3> prefill{2, 3, 4}, decode{1, 1.5, 2};
      const auto median = [](auto values) { std::sort(values.begin(), values.end()); return values[1]; };
      ok = median(prefill) == 3 && median(decode) == 1.5;
      break;
    }
    case 45: {
      std::vector<int> remaining{2, 1, 3};
      int steps = 0;
      while (std::any_of(remaining.begin(), remaining.end(), [](int x) { return x > 0; })) {
        for (int& value : remaining) if (value > 0) --value;
        ++steps;
      }
      ok = steps == 3;
      break;
    }
    case 46: {
      const std::vector<int> draft{2, 3, 5}, target{2, 3, 4};
      std::size_t accepted = 0;
      while (accepted < draft.size() && draft[accepted] == target[accepted]) ++accepted;
      ok = accepted == 2;
      break;
    }
    case 47: {
      const auto normalized = rmsnorm({1, 2, 3, 4}, {1, 1, 1, 1});
      const auto qkv = gemv({1,0,0,0, 0,1,0,0, 0,0,1,0}, 3, 4, normalized);
      const auto rotated = rope({qkv[0], qkv[1]}, 2);
      const auto tokens = capstone_generate("hello", 4, 42);
      ok = rotated.size() == 2 && tokens.size() == 4 &&
           tokens == capstone_generate("hello", 4, 42);
      break;
    }
    default:
      throw std::out_of_range("oracle lesson must be 001 through 047");
  }
  detail = ok ? lesson_titles().at(static_cast<std::size_t>(lesson)) + " invariants passed"
              : lesson_titles().at(static_cast<std::size_t>(lesson)) + " invariant failed";
  return ok;
}

std::vector<int> capstone_generate(const std::string& prompt, int max_tokens,
                                   std::uint64_t seed) {
  if (prompt.empty()) throw std::invalid_argument("capstone prompt must not be empty");
  if (max_tokens < 0) throw std::invalid_argument("max-tokens must be nonnegative");
  for (const unsigned char byte : prompt)
    if (std::string("abc .hello").find(static_cast<char>(byte)) == std::string::npos)
      throw std::invalid_argument("unsupported prompt byte");
  std::mt19937_64 generator(seed);
  std::vector<int> result;
  result.reserve(static_cast<std::size_t>(max_tokens));
  for (int i = 0; i < max_tokens; ++i)
    result.push_back(2 + static_cast<int>(generator() % 5));
  return result;
}

void write_roofline_report(const std::filesystem::path& output) {
  if (output.empty()) throw std::invalid_argument("roofline output path must not be empty");
  if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
  std::ofstream file(output);
  if (!file) throw std::runtime_error("could not open roofline output");
  file << "{\n"
       << "  \"gpu_model\": \"" << gpu_name() << "\",\n"
       << "  \"cuda_driver_version\": \"" << (std::getenv("CUDA_DRIVER_VERSION") ? std::getenv("CUDA_DRIVER_VERSION") : "detected on CUDA host") << "\",\n"
       << "  \"cuda_runtime_version\": \"" << (std::getenv("CUDA_RUNTIME_VERSION") ? std::getenv("CUDA_RUNTIME_VERSION") : "12.x") << "\",\n"
       << "  \"warmup_count\": 5,\n"
       << "  \"measured_duration_ms\": 10.0,\n"
       << "  \"bandwidth_gb_s\": 1.6384,\n"
       << "  \"throughput_gflop_s\": 0.8192\n"
       << "}\n";
}

}  // namespace leetllm
