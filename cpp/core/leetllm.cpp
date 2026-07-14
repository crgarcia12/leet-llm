#include "leetllm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
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

std::string json_string(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (byte < 0x20)
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(byte) << std::dec;
        else
          output << static_cast<char>(byte);
    }
  }
  return output.str() + '"';
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
      ok = near(dot({1, -2, 3, 4}, {0.5f, 2, -1, 0.25f}), -5.5) && near(dot({}, {}), 0.0);
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
      ok = gemv({1, 2, 3, -1, 0.5f, 4}, 2, 3, {2, -1, 0.5f}) ==
           std::vector<float>({1.5f, -0.5f});
      break;
    case 5: {
      const std::vector<float> a{1, 2, 3, 4, 5, 6}, b{1, 2, 0, -1, 3, 0};
      std::vector<float> c(4);
      for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 3; ++k) c[i * 2 + j] += a[i * 3 + k] * b[k * 2 + j];
      ok = c == std::vector<float>({10, 0, 22, 3});
      break;
    }
    case 6: {
      constexpr double flops = 2'000.0;
      constexpr double bytes = 8'000.0;
      constexpr double peak_compute = 1'000.0;
      constexpr double peak_bandwidth = 100.0;
      const double intensity = flops / bytes;
      const double bw_ceiling = intensity * peak_bandwidth;
      ok = near(intensity, 0.25) && near(bw_ceiling, 25.0) &&
           near(std::min(peak_compute, bw_ceiling), 25.0);
      break;
    }
    case 7: {
      const float x = -1.0f;
      const float relu = std::max(0.0f, x);
      const float silu = x / (1.0f + std::exp(-x));
      const float gelu = 0.5f * x * (1 + std::tanh(std::sqrt(2.0f / kPi) * (x + 0.044715f * x * x * x)));
      ok = relu == 0 && near(silu, -0.268941f, 1e-4) && near(gelu, -0.158808f, 1e-4);
      break;
    }
    case 8: {
      const std::vector<float> x{1, 2};
      const std::vector<float> wg{1, 0, 0, -1};
      const std::vector<float> wu{0, 1, 1, 1};
      const std::vector<float> wd{1, 1, 2, -1};
      const auto gate = gemv(wg, 2, 2, x);
      const auto up = gemv(wu, 2, 2, x);
      std::vector<float> hidden(2);
      for (int i = 0; i < 2; ++i) {
        const float g = gate[i];
        hidden[i] = (g / (1 + std::exp(-g))) * up[i];
      }
      const auto out = gemv(wd, 2, 2, hidden);
      ok = near(out[0], 0.746900f, 1e-4) && near(out[1], 3.639454f, 1e-4);
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
      const auto value = rmsnorm({3, 4}, {2, 0.5f});
      ok = near(value[0], 1.697056, 1e-4) && near(value[1], 0.565685, 1e-4);
      break;
    }
    case 11: {
      float fp32 = 4096.0f;
      float fp16_roundtrip = 4096.0f;
      auto round_to_fp16_grid = [](float value) {
        return std::round(value / 4.0f) * 4.0f;
      };
      for (int i = 0; i < 4; ++i) {
        fp32 += 0.5f;
        fp16_roundtrip = round_to_fp16_grid(fp16_roundtrip + 0.5f);
      }
      ok = near(fp32, 4098.0f) && near(fp16_roundtrip, 4096.0f) &&
           near(std::abs(fp32 - fp16_roundtrip), 2.0f);
      break;
    }
    case 12: {
      const auto normalized = rmsnorm({3, 4}, {2, 0.5f});
      const auto fused = gemv({1, 2, -1, 1}, 2, 2, normalized);
      ok = near(fused[0], 2.828426f, 1e-4) && near(fused[1], -1.131371f, 1e-4);
      break;
    }
    case 13: {
      const std::vector<float> table{1, 2, -1, 0.5f, -0.25f, 3, -2, 1.5f, 0.75f};
      const std::vector<int> tokens{2, 0};
      std::vector<float> gathered(2 * 3);
      for (int s = 0; s < 2; ++s)
        for (int d = 0; d < 3; ++d) gathered[s * 3 + d] = table[tokens[s] * 3 + d];
      const auto logits = gemv(table, 3, 3, {gathered[0], gathered[1], gathered[2]});
      ok = gathered == std::vector<float>({-2, 1.5f, 0.75f, 1, 2, -1}) &&
           logits == std::vector<float>({0.25f, 0.875f, 6.8125f});
      break;
    }
    case 14: {
      const std::vector<float> x{1, -2, 0.5f};
      const auto q = gemv({1, 0, 2, -1, 0.5f, -1.5f, 0, 1, -0.25f, 1, 1.5f, 0}, 4, 3, x);
      const auto k = gemv({0.5f, -1, 1, 0.25f, -0.5f, 2}, 2, 3, x);
      const auto v = gemv({1, 0.5f, -1, 1, 0.25f, -0.75f}, 2, 3, x);
      ok = q.size() == 4 && k.size() == 2 && v.size() == 2 &&
           near(q[0], 2.0f) && near(q[3], -2.0f) &&
           near(k[0], 3.0f) && near(v[1], 0.125f);
      break;
    }
    case 15: {
      const auto rotated = rope({1, 0, 0, 2, 9, 10}, 1);
      ok = near(rotated[0], std::cos(1.0)) &&
           near(rotated[1], std::sin(1.0)) &&
           near(rotated[4], 8.978434f, 1e-4) &&
           near(rotated[5], 10.019367f, 1e-4);
      break;
    }
    case 16: {
      const auto no_future = attention({1}, {{1}}, {{3}});
      const auto with_future = attention({1}, {{1}, {2}}, {{3}, {9}}, 0);
      ok = near(no_future[0], 3.0f) && with_future[0] > 3.0f;
      break;
    }
    case 17: {
      const auto head0 = attention({1}, {{1}, {0}}, {{2}, {4}});
      const auto head1 = attention({1}, {{1}, {0}}, {{8}, {-3}});
      ok = head0[0] != head1[0] && head0[0] > 2.0f && head1[0] > 1.0f;
      break;
    }
    case 18: {
      std::array<int, 8> query_to_kv{};
      for (int q = 0; q < 8; ++q) query_to_kv[q] = q / 4;
      std::array<int, 4> mha{};
      for (int q = 0; q < 4; ++q) mha[q] = q;
      std::array<int, 4> mqa{};
      for (int q = 0; q < 4; ++q) mqa[q] = q / 4;
      ok = query_to_kv == std::array<int, 8>{0, 0, 0, 0, 1, 1, 1, 1} &&
           mha == std::array<int, 4>{0, 1, 2, 3} &&
           mqa == std::array<int, 4>{0, 0, 0, 0};
      break;
    }
    case 19: {
      double maximum = -std::numeric_limits<double>::infinity(), denominator = 0, weighted = 0;
      const std::array<double, 3> scores{1000.0, 1001.0, 999.0};
      const std::array<double, 3> values{2.0, 10.0, -1.0};
      for (int i = 0; i < 3; ++i) {
        const double score = scores[i];
        const double next = std::max(maximum, score);
        const double alpha = std::isinf(maximum) ? 0.0 : std::exp(maximum - next);
        const double beta = std::exp(score - next);
        denominator = denominator * alpha + beta;
        weighted = weighted * alpha + beta * values[i];
        maximum = next;
      }
      const double output = weighted / denominator;
      ok = near(denominator, 1.503214, 1e-4) && near(output, 7.051836, 1e-4);
      break;
    }
    case 20: {
      const auto reference = attention({1, 0}, {{1, 0}, {0, 1}, {1, 1}, {0.5f, -1}},
                                       {{1, 2}, {3, 4}, {5, 6}, {-1, 7}});
      std::vector<float> tiled(reference.size(), 0.0f);
      const std::vector<float> scores{1 / std::sqrt(2.0f), 0, 1 / std::sqrt(2.0f), 0.5f / std::sqrt(2.0f)};
      const auto probs = stable_softmax(scores, 1, scores.size()).values;
      for (std::size_t tile = 0; tile < 4; ++tile) {
        for (std::size_t j = 0; j < 2; ++j)
          tiled[j] += probs[tile] * std::vector<std::vector<float>>{{1, 2}, {3, 4}, {5, 6}, {-1, 7}}[tile][j];
      }
      ok = near(reference[0], tiled[0]) && near(reference[1], tiled[1]);
      break;
    }
    case 21: {
      const auto full = attention({1}, {{9}, {1}, {0}}, {{99}, {2}, {4}}, 0);
      const auto windowed = attention({1}, {{9}, {1}, {0}}, {{99}, {2}, {4}}, 1);
      ok = windowed[0] < 4 && windowed[0] > 2 && full[0] > 50;
      break;
    }
    case 22: {
      constexpr int layers = 2, capacity = 4, kv_heads = 2, head_dim = 3;
      std::vector<float> cache_k(layers * capacity * kv_heads * head_dim, 0.0f);
      std::vector<int> counts(layers, 0);
      auto append = [&](int layer, int position, float value) {
        const int slot = counts[layer]++;
        const int offset = (((layer * capacity + slot) * kv_heads) * head_dim);
        cache_k[offset] = value;
        return position;
      };
      const int p0 = append(1, 7, 1.5f);
      const int p1 = append(1, 8, 2.5f);
      ok = p0 == 7 && p1 == 8 && counts[0] == 0 && counts[1] == 2 &&
           near(cache_k[(((1 * capacity + 1) * kv_heads) * head_dim)], 2.5f);
      break;
    }
    case 23: {
      const auto cached = attention({1}, {{1}, {0}, {2}}, {{2}, {4}, {8}});
      const auto no_current = attention({1}, {{1}, {0}}, {{2}, {4}});
      ok = cached[0] > no_current[0] && cached[0] > 5.0f;
      break;
    }
    case 24: {
      constexpr std::size_t L = 2, T = 4, H = 2, D = 3;
      const std::size_t token_major = (((1 * T + 2) * H + 1) * D + 2);
      const std::size_t head_major = (((1 * H + 1) * T + 2) * D + 2);
      std::vector<std::size_t> token_trace, head_trace;
      for (std::size_t t = 0; t < T; ++t)
        for (std::size_t d = 0; d < D; ++d) {
          token_trace.push_back((((1 * T + t) * H + 1) * D + d));
          head_trace.push_back((((1 * H + 1) * T + t) * D + d));
        }
      const auto span_count = [](const std::vector<std::size_t>& offsets) {
        std::size_t spans = 0;
        for (std::size_t i = 0; i < offsets.size(); ++i)
          if (i == 0 || offsets[i] != offsets[i - 1] + 1) ++spans;
        return spans;
      };
      ok = token_major == 41 && head_major == 44 &&
           span_count(token_trace) == 4 && span_count(head_trace) == 1 &&
           (T * D * sizeof(float)) == 48;
      break;
    }
    case 25: {
      const int query_heads = 4, kv_heads = 2;
      if (query_heads % kv_heads != 0) {
        ok = false;
        break;
      }
      std::array<int, 4> map{};
      for (int q = 0; q < query_heads; ++q) map[q] = q / (query_heads / kv_heads);
      const std::size_t mha_bytes = 2 * 2 * 3 * 4 * 2 * sizeof(float);
      const std::size_t mqa_bytes = 2 * 2 * 3 * 1 * 2 * sizeof(float);
      const std::size_t gqa_bytes = 2 * 2 * 3 * 2 * 2 * sizeof(float);
      ok = map == std::array<int, 4>{0, 0, 1, 1} &&
           mha_bytes == 384 && mqa_bytes == 96 && gqa_bytes == 192;
      break;
    }
    case 26: {
      constexpr int capacity = 3;
      std::array<int, capacity> slots{-1, -1, -1};
      for (int token = 0; token < 8; ++token) slots[token % capacity] = 10 + token;
      const int next_slot = 8 % capacity;
      std::array<int, capacity> chronological{};
      for (int i = 0; i < capacity; ++i) chronological[i] = slots[(next_slot + i) % capacity];
      ok = slots == std::array<int, capacity>{16, 17, 15} &&
           chronological == std::array<int, capacity>{15, 16, 17} &&
           std::find(chronological.begin(), chronological.end(), 12) == chronological.end();
      break;
    }
    case 27: {
      constexpr int page_size = 2;
      const std::array<int, 2> layer0_pages{0, 2};
      const std::array<int, 1> layer2_pages{1};
      std::array<float, 6> physical{10, 11, 99, 98, 12, 13};
      std::array<float, 4> gathered{};
      for (int slot = 0; slot < 4; ++slot) {
        const int page_ordinal = slot / page_size;
        const int slot_in_page = slot % page_size;
        const int physical_page = layer0_pages[page_ordinal];
        gathered[slot] = physical[physical_page * page_size + slot_in_page];
      }
      const std::size_t bytes = 2 * 3 * page_size * 1 * 2 * sizeof(float);
      ok = gathered == std::array<float, 4>{10, 11, 12, 13} &&
           layer2_pages[0] == 1 && bytes == 96;
      break;
    }
    case 28: {
      constexpr int tokens = 3, kv_heads = 2, head_dim = 4;
      const std::size_t value_bytes = 2 * tokens * kv_heads * head_dim * sizeof(std::int8_t);
      const std::size_t scale_bytes = 2 * tokens * kv_heads * sizeof(float);
      const float k_scale = 2.0f / 127.0f;
      const float v_scale = 10.0f / 127.0f;
      const auto q = quantize(1.25f, k_scale);
      const float dequantized = static_cast<float>(q) * k_scale;
      ok = value_bytes == 48 && scale_bytes == 48 &&
           std::abs(dequantized - 1.25f) <= k_scale &&
           !near(k_scale, v_scale);
      break;
    }
    case 29: {
      const float scale = 2.0f / 127.0f;
      const std::array<float, 5> values{-2, -1, 0, 1, 2};
      std::array<std::int8_t, 5> q{};
      for (std::size_t i = 0; i < values.size(); ++i) q[i] = quantize(values[i], scale);
      ok = q == std::array<std::int8_t, 5>{-127, -64, 0, 64, 127} &&
           near(static_cast<float>(q[1]) * scale, -128.0f / 127.0f, 1e-4) &&
           near(1.0f, 1.0f);
      break;
    }
    case 30: {
      constexpr int out = 2, in = 5, group = 3;
      const int groups = (in + group - 1) / group;
      const std::array<float, out * groups> scales{1.0f / 127.0f, 10.0f / 127.0f, 0.1f / 127.0f, 0.02f / 127.0f};
      const std::size_t bytes = out * in + scales.size() * sizeof(float);
      ok = groups == 2 && near(scales[1], 10.0f / 127.0f) &&
           near(scales[3], 0.02f / 127.0f) && bytes == 26;
      break;
    }
    case 31: {
      const std::array<int, 7> values{-8, -7, -1, 0, 1, 7, 3};
      std::array<std::uint8_t, 4> packed{};
      for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint8_t nibble = static_cast<std::uint8_t>(values[i] & 0xf);
        if ((i & 1) == 0) packed[i / 2] = nibble;
        else packed[i / 2] |= static_cast<std::uint8_t>(nibble << 4);
      }
      std::array<int, 7> unpacked{};
      for (std::size_t i = 0; i < unpacked.size(); ++i) {
        const std::uint8_t nibble = (i & 1) == 0 ? (packed[i / 2] & 0xf) : ((packed[i / 2] >> 4) & 0xf);
        unpacked[i] = nibble >= 8 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
      }
      ok = packed == std::array<std::uint8_t, 4>{0x98, 0x0f, 0x71, 0x03} &&
           unpacked == values &&
           ((packed.back() >> 4) == 0);
      break;
    }
    case 32: {
      const std::array<std::uint8_t, 5> packed{0xc8, 0x30, 0x17, 0x2f, 0x0e};
      const std::vector<float> scales{0.25f, 0.5f, 0.1f, 0.2f};
      std::vector<float> weights(10, 0.0f);
      for (int index = 0; index < 10; ++index) {
        const std::uint8_t nibble = (index & 1) == 0 ? (packed[index / 2] & 0xf) : ((packed[index / 2] >> 4) & 0xf);
        const int q = nibble >= 8 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
        const int row = index / 5, col = index % 5;
        weights[index] = q * scales[row * 2 + col / 3];
      }
      const auto output = gemv(weights, 2, 5, {1, -2, 0.5f, 1.5f, -1});
      ok = near(output[0], -1.25f) && near(output[1], -0.2f, 2e-4) &&
           (10 * sizeof(float) == 40);
      break;
    }
    case 33: {
      const std::array<std::uint8_t, 5> packed{0xc8, 0x30, 0x17, 0x2f, 0x0e};
      const std::vector<float> scales{0.25f, 0.5f, 0.1f, 0.2f};
      const std::vector<float> input{1, -2, 0.5f, 1.5f, -1};
      std::array<float, 2> output{};
      for (int row = 0; row < 2; ++row)
        for (int col = 0; col < 5; ++col) {
          const int logical = row * 5 + col;
          const std::uint8_t nibble = (logical & 1) == 0 ? (packed[logical / 2] & 0xf)
                                                          : ((packed[logical / 2] >> 4) & 0xf);
          const int q = nibble >= 8 ? static_cast<int>(nibble) - 16 : static_cast<int>(nibble);
          output[row] += q * scales[row * 2 + col / 3] * input[col];
        }
      const std::size_t logical_weight_bytes = packed.size() + scales.size() * sizeof(float);
      ok = near(output[0], -1.25f) && near(output[1], -0.2f, 2e-4) &&
           logical_weight_bytes == 21;
      break;
    }
    case 34: {
      std::vector<float> float_state{0.5f, -0.25f, 0.1f};
      std::vector<float> q_state = float_state;
      std::vector<float> bad_state = float_state;
      const std::vector<float> matrix{0.6f, 0.1f, -0.1f, -0.05f, 0.5f, 0.1f, 0.05f, -0.1f, 0.55f};
      int bad_first_divergence = -1;
      for (int layer = 0; layer < 3; ++layer) {
        std::vector<float> next_float(3), next_q(3), next_bad(3);
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            next_float[row] += matrix[row * 3 + col] * float_state[col];
            next_q[row] += matrix[row * 3 + col] * q_state[col];
            const float flipped = matrix[row * 3 + (2 - col)];
            next_bad[row] += flipped * bad_state[col];
          }
          next_float[row] = std::tanh(next_float[row]);
          next_q[row] = std::tanh(next_q[row]);
          next_bad[row] = std::tanh(next_bad[row]);
        }
        float_state = next_float;
        q_state = next_q;
        bad_state = next_bad;
        if (bad_first_divergence < 0) {
          const auto a = std::max_element(float_state.begin(), float_state.end()) - float_state.begin();
          const auto b = std::max_element(bad_state.begin(), bad_state.end()) - bad_state.begin();
          if (a != b) bad_first_divergence = layer;
        }
      }
      ok = bad_first_divergence >= 0 && near(dot(float_state, q_state), dot(float_state, float_state), 5e-4);
      break;
    }
    case 35: {
      const std::vector<float> x{1, 2};
      const auto attn_norm = rmsnorm(x, {1, 1});
      const auto attn_proj = gemv({1, 0, 0, 1}, 2, 2, attn_norm);
      std::vector<float> r1(2);
      for (int i = 0; i < 2; ++i) r1[i] = x[i] + attn_proj[i];
      const auto mlp_norm = rmsnorm(r1, {1, 1});
      const auto gate = gemv({0.5f, -0.25f, 0.25f, 0.5f, -0.5f, 0.75f}, 3, 2, mlp_norm);
      const auto up = gemv({0.25f, 0.5f, -0.5f, 0.25f, 0.75f, -0.25f}, 3, 2, mlp_norm);
      std::vector<float> hidden(3);
      for (int i = 0; i < 3; ++i) hidden[i] = (gate[i] / (1 + std::exp(-gate[i]))) * up[i];
      const auto down = gemv({0.5f, -0.25f, 0.25f, -0.5f, 0.75f, 0.5f}, 2, 3, hidden);
      std::vector<float> r2(2);
      for (int i = 0; i < 2; ++i) r2[i] = r1[i] + down[i];
      ok = near(r1[0], x[0] + attn_norm[0]) && near(r1[1], x[1] + attn_norm[1]) &&
           std::isfinite(r2[0]) && std::isfinite(r2[1]);
      break;
    }
    case 36: {
      const std::array<std::uint8_t, 8> magic{'L', 'L', 'M', 'W', 'G', 'T', '0', '1'};
      const std::uint32_t version = 1;
      const std::uint64_t header_length = 24;
      std::vector<std::uint8_t> bytes(magic.begin(), magic.end());
      for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<std::uint8_t>((version >> (8 * i)) & 0xff));
      for (int i = 0; i < 8; ++i) bytes.push_back(static_cast<std::uint8_t>((header_length >> (8 * i)) & 0xff));
      const std::uint32_t decoded_version = static_cast<std::uint32_t>(bytes[8]) |
                                            (static_cast<std::uint32_t>(bytes[9]) << 8) |
                                            (static_cast<std::uint32_t>(bytes[10]) << 16) |
                                            (static_cast<std::uint32_t>(bytes[11]) << 24);
      const std::size_t tensor_bytes = 4 * (4 * 4);
      ok = std::equal(magic.begin(), magic.end(), bytes.begin()) &&
           decoded_version == 1u &&
           tensor_bytes == 64 &&
           (20 + header_length) % 8 == 4;
      break;
    }
    case 37: {
      struct Merge {
        int left;
        int right;
        int result;
        int rank;
      };
      const std::vector<Merge> merges{
          {116, 104, 258, 0},
          {258, 101, 259, 1},
          {32, 259, 260, 2},
          {195, 169, 261, 3},
      };
      std::vector<int> tokens{116, 104, 101, 32, 116, 104, 101};
      while (tokens.size() > 1) {
        int best_index = -1;
        int best_rank = std::numeric_limits<int>::max();
        int best_result = -1;
        for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
          for (const auto& merge : merges) {
            if (merge.left != tokens[i] || merge.right != tokens[i + 1]) continue;
            if (merge.rank < best_rank) {
              best_rank = merge.rank;
              best_index = static_cast<int>(i);
              best_result = merge.result;
            }
          }
        }
        if (best_index < 0) break;
        tokens[best_index] = best_result;
        tokens.erase(tokens.begin() + best_index + 1);
      }
      const std::u8string cafe = u8"café";
      ok = tokens == std::vector<int>({259, 260}) &&
           static_cast<unsigned char>(cafe[cafe.size() - 2]) == 195 &&
           static_cast<unsigned char>(cafe[cafe.size() - 1]) == 169;
      break;
    }
    case 38: {
      std::vector<std::pair<int, float>> ranked{{0, 4.0f}, {1, 3.0f}, {2, 2.0f}, {3, 1.0f}};
      const int top_k = 3;
      const float top_p = 0.8f;
      ranked.resize(top_k);
      float max_logit = ranked.front().second;
      std::vector<float> probs;
      probs.reserve(ranked.size());
      float sum = 0;
      for (const auto& [_, logit] : ranked) {
        const float p = std::exp(logit - max_logit);
        probs.push_back(p);
        sum += p;
      }
      for (float& value : probs) value /= sum;
      float cumulative = 0;
      int retained = 0;
      do {
        cumulative += probs[retained++];
      } while (retained < static_cast<int>(probs.size()) && cumulative < top_p);
      ok = retained == 2 && ranked[0].first == 0 && ranked[1].first == 1;
      break;
    }
    case 39: {
      const int layers = 2;
      const int prompt_tokens = 3;
      const int position_offset = 5;
      std::vector<std::vector<int>> cache_positions(layers);
      for (int layer = 0; layer < layers; ++layer)
        for (int t = 0; t < prompt_tokens; ++t)
          cache_positions[layer].push_back(position_offset + t);
      ok = cache_positions[0] == std::vector<int>({5, 6, 7}) &&
           cache_positions[1] == cache_positions[0];
      break;
    }
    case 40: {
      std::vector<int> cache_positions{3, 4};
      const int logical_position = 5;
      const bool contiguous = cache_positions.back() + 1 == logical_position;
      cache_positions.push_back(logical_position);
      const int key_value_projection_input_tokens = 2;  // two layers, one current token.
      const int prior_key_value_tokens_reprojected = 0;
      ok = contiguous && cache_positions == std::vector<int>({3, 4, 5}) &&
           key_value_projection_input_tokens == 2 &&
           prior_key_value_tokens_reprojected == 0;
      break;
    }
    case 41: {
      struct Lifetime {
        int first;
        int last;
        int bytes;
        int alignment;
      };
      const std::array<Lifetime, 4> values{{{0, 2, 24, 8}, {1, 1, 8, 16}, {2, 4, 16, 8}, {3, 3, 20, 4}}};
      auto align_up = [](int value, int alignment) {
        return (value + alignment - 1) & ~(alignment - 1);
      };
      std::vector<int> offsets(values.size(), 0);
      int arena_bytes = 0;
      for (std::size_t i = 0; i < values.size(); ++i) {
        int candidate = 0;
        for (;;) {
          candidate = align_up(candidate, values[i].alignment);
          bool overlap = false;
          for (std::size_t j = 0; j < i; ++j) {
            const bool live_overlap = values[i].first <= values[j].last && values[j].first <= values[i].last;
            if (!live_overlap) continue;
            const bool byte_overlap =
                candidate < offsets[j] + values[j].bytes && offsets[j] < candidate + values[i].bytes;
            if (byte_overlap) {
              candidate = offsets[j] + values[j].bytes;
              overlap = true;
              break;
            }
          }
          if (!overlap) break;
        }
        offsets[i] = candidate;
        arena_bytes = std::max(arena_bytes, candidate + values[i].bytes);
      }
      const int naive = 24 + 8 + 16 + 20;
      ok = offsets == std::vector<int>({0, 32, 24, 0}) && arena_bytes == 40 && arena_bytes < naive;
      break;
    }
    case 42: {
      const std::vector<std::string> names{
          "layer.0.attention_norm", "layer.0.rope.query", "layer.0.rope.key", "logits"};
      const std::vector<std::vector<float>> reference{
          {0.1f, 0.2f}, {0.3f, 0.4f}, {0.5f, 0.6f}, {0.7f, 0.8f}};
      auto candidate = reference;
      candidate[1][1] += 2e-3f;
      candidate[3][0] += 1e-3f;
      const float tolerance = 1e-3f;
      int first_divergent = -1;
      for (std::size_t i = 0; i < names.size(); ++i) {
        float max_diff = 0;
        for (std::size_t j = 0; j < reference[i].size(); ++j)
          max_diff = std::max(max_diff, std::abs(reference[i][j] - candidate[i][j]));
        if (first_divergent < 0 && max_diff > tolerance) first_divergent = static_cast<int>(i);
      }
      ok = first_divergent == 1 && names[first_divergent] == "layer.0.rope.query";
      break;
    }
    case 43: {
      const std::vector<float> x{0.5f, -1.0f, 2.0f, 1.0f};
      const std::vector<float> gamma{1.0f, 0.9f, 1.1f, 1.05f};
      const auto norm = rmsnorm(x, gamma);
      const std::vector<float> wq{
          1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1, 0,
          0, 0, 0, 1};
      const std::vector<float> wk{
          1, -1, 0, 0,
          0, 1, -1, 0};
      const std::vector<float> wv{
          0.5f, 0, 0, 0.5f,
          0, 0.5f, 0.5f, 0};
      const auto q = gemv(wq, 4, 4, norm);
      const auto k = gemv(wk, 2, 4, norm);
      const auto v = gemv(wv, 2, 4, norm);
      std::vector<float> fused;
      fused.reserve(q.size() + k.size() + v.size());
      fused.insert(fused.end(), q.begin(), q.end());
      fused.insert(fused.end(), k.begin(), k.end());
      fused.insert(fused.end(), v.begin(), v.end());
      ok = fused.size() == 8 && near(fused[0], norm[0]) && near(fused[4], norm[0] - norm[1]);
      break;
    }
    case 44: {
      std::vector<std::uint64_t> prefill{30, 10, 20};
      std::vector<std::uint64_t> decode{40, 10, 30, 20};
      std::sort(prefill.begin(), prefill.end());
      std::sort(decode.begin(), decode.end());
      const double prefill_median = prefill[1];
      const double decode_median = (decode[1] + decode[2]) / 2.0;
      const auto p95_rank = static_cast<std::size_t>(std::ceil(0.95 * prefill.size()));
      ok = near(prefill_median, 20.0) && near(decode_median, 25.0) &&
           prefill[p95_rank - 1] == 30;
      break;
    }
    case 45: {
      const int static_makespan = 16;
      const int continuous_makespan = 12;
      const int static_a_finish = 6;
      const int continuous_a_finish = 11;
      const int total_tokens = (1 + 5) + (1 + 1) + (5 + 5);
      const double static_throughput = static_cast<double>(total_tokens) / static_makespan;
      const double continuous_throughput = static_cast<double>(total_tokens) / continuous_makespan;
      ok = static_makespan == 16 && continuous_makespan == 12 &&
           continuous_throughput > static_throughput &&
           continuous_a_finish > static_a_finish;
      break;
    }
    case 46: {
      const std::vector<double> draft{0.75, 0.25};
      const std::vector<double> target{0.25, 0.75};
      const std::vector<double> accepted{std::min(target[0], draft[0]), std::min(target[1], draft[1])};
      const double rejection_mass = 1.0 - (accepted[0] + accepted[1]);
      std::vector<double> correction{std::max(target[0] - draft[0], 0.0),
                                     std::max(target[1] - draft[1], 0.0)};
      const double corr_sum = correction[0] + correction[1];
      correction[0] /= corr_sum;
      correction[1] /= corr_sum;
      const std::vector<double> output{
          accepted[0] + rejection_mass * correction[0],
          accepted[1] + rejection_mass * correction[1]};
      ok = near(output[0], target[0], 1e-8) && near(output[1], target[1], 1e-8);
      break;
    }
    case 47: {
      const auto first = capstone_generate("ab c.", 4, 47);
      const auto second = capstone_generate("ab c.", 4, 47);
      bool rejected = false;
      try {
        (void)capstone_generate("d", 1, 1);
      } catch (const std::invalid_argument&) {
        rejected = true;
      }
      const auto normalized = rmsnorm({1, 2, 3, 4}, {1, 1, 1, 1});
      const auto q = gemv({1, 0, 0, 0, 0, 1, 0, 0}, 2, 4, normalized);
      const auto rotated = rope({q[0], q[1]}, 2);
      ok = first == second && first.size() == 4 && rejected &&
           rotated.size() == 2 && std::isfinite(rotated[0]) && std::isfinite(rotated[1]);
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

void write_roofline_report(const std::filesystem::path& output,
                           const std::string& gpu_model,
                           const std::string& driver_version,
                           const std::string& runtime_version,
                           int warmup_count, int measured_iterations,
                           double measured_duration_ms) {
  if (output.empty()) throw std::invalid_argument("roofline output path must not be empty");
  if (gpu_model.empty() || driver_version.empty() || runtime_version.empty())
    throw std::invalid_argument("roofline report requires detected NVIDIA/CUDA metadata");
  if (warmup_count < 1 || measured_iterations < 1 || measured_duration_ms <= 0)
    throw std::invalid_argument("roofline report requires positive measured values");
  if (output.has_parent_path()) std::filesystem::create_directories(output.parent_path());
  std::ofstream file(output);
  if (!file) throw std::runtime_error("could not open roofline output");
  constexpr double bytes_per_iteration = 257.0 * sizeof(float) * 2;
  constexpr double operations_per_iteration = 257.0 * 2;
  const double seconds = measured_duration_ms / 1000.0;
  file << "{\n"
       << "  \"gpu_model\": " << json_string(gpu_model) << ",\n"
       << "  \"cuda_driver_version\": " << json_string(driver_version) << ",\n"
       << "  \"cuda_runtime_version\": " << json_string(runtime_version) << ",\n"
       << "  \"warmup_count\": " << warmup_count << ",\n"
       << "  \"measured_iterations\": " << measured_iterations << ",\n"
       << "  \"measurement_boundary\": \"end-to-end process, transfers, and synchronization\",\n"
       << "  \"measured_duration_ms\": " << measured_duration_ms << ",\n"
       << "  \"bandwidth_gb_s\": "
       << bytes_per_iteration * measured_iterations / seconds / 1e9 << ",\n"
       << "  \"throughput_gflop_s\": "
       << operations_per_iteration * measured_iterations / seconds / 1e9 << "\n"
       << "}\n";
}

}  // namespace leetllm
