#include "cuda_check.hpp"

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

struct Request {
  std::string id;
  int arrival;
  int prompt_tokens;
  int decode_tokens;
  int remaining;
  int finish = -1;
  int previous_token;
  int salt;
};

__global__ void next_token_kernel(const int* previous, const int* salts, int step,
                                  int* token_out, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) token_out[i] = (salts[i] + previous[i] * 17 + step * 13) % 256;
}

int salt_for(const std::string& id) {
  int salt = 0;
  for (unsigned char c : id) salt = (salt * 31 + c) % 251;
  return salt;
}

int simulate(bool continuous) {
  std::vector<Request> reqs{{"A", 0, 1, 5, 5, -1, 10, salt_for("A")},
                            {"B", 0, 1, 1, 1, -1, 20, salt_for("B")},
                            {"C", 1, 5, 5, 5, -1, 34, salt_for("C")}};
  const int slots = 2;
  int time = 0;
  int completed = 0;
  std::vector<int> active;
  int next_unqueued = 0;

  auto enqueue_static = [&]() {
    while (next_unqueued < static_cast<int>(reqs.size()) &&
           reqs[next_unqueued].arrival <= time && static_cast<int>(active.size()) < slots) {
      active.push_back(next_unqueued++);
    }
  };

  enqueue_static();
  time += std::max(reqs[active[0]].prompt_tokens, reqs[active[1]].prompt_tokens);

  int decode_step = 0;
  while (completed < static_cast<int>(reqs.size())) {
    if (active.empty()) {
      time = std::max(time, reqs[next_unqueued].arrival);
      enqueue_static();
      if (!active.empty()) {
        int max_prompt = 0;
        for (int idx : active) max_prompt = std::max(max_prompt, reqs[idx].prompt_tokens);
        time += max_prompt;
      }
      continue;
    }

    std::vector<int> previous(active.size()), salts(active.size()), generated(active.size());
    for (int i = 0; i < static_cast<int>(active.size()); ++i) {
      previous[i] = reqs[active[i]].previous_token;
      salts[i] = reqs[active[i]].salt;
    }
    int *d_prev = nullptr, *d_salts = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prev, previous.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_salts, salts.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out, generated.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_prev, previous.data(), previous.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_salts, salts.data(), salts.size() * sizeof(int), cudaMemcpyHostToDevice));
    next_token_kernel<<<1, 128>>>(d_prev, d_salts, decode_step, d_out, static_cast<int>(generated.size()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(generated.data(), d_out, generated.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_salts));
    CUDA_CHECK(cudaFree(d_out));

    time += 1;
    ++decode_step;
    std::vector<int> survivors;
    for (int i = 0; i < static_cast<int>(active.size()); ++i) {
      Request& r = reqs[active[i]];
      r.previous_token = generated[i];
      --r.remaining;
      if (r.remaining == 0) {
        r.finish = time;
        ++completed;
      } else {
        survivors.push_back(active[i]);
      }
    }
    active = survivors;

    if (continuous && next_unqueued < static_cast<int>(reqs.size()) &&
        static_cast<int>(active.size()) < slots && reqs[next_unqueued].arrival <= time) {
      active.push_back(next_unqueued++);
      int max_prompt = 0;
      for (int idx : active) max_prompt = std::max(max_prompt, reqs[idx].prompt_tokens);
      time += max_prompt;
    } else if (!continuous && active.empty() && next_unqueued < static_cast<int>(reqs.size())) {
      time = std::max(time, reqs[next_unqueued].arrival);
      enqueue_static();
      int max_prompt = 0;
      for (int idx : active) max_prompt = std::max(max_prompt, reqs[idx].prompt_tokens);
      time += max_prompt;
    }
  }
  if (!continuous) {
    if (!(reqs[0].finish == 6 && reqs[1].finish == 2 && reqs[2].finish == 16)) return -1;
  } else {
    if (!(reqs[0].finish == 11 && reqs[1].finish == 2 && reqs[2].finish == 12)) return -1;
  }
  return time;
}

int main() {
  const int static_makespan = simulate(false);
  const int continuous_makespan = simulate(true);
  if (static_makespan != 16 || continuous_makespan != 12) return 1;
  std::cout << "p045 static/continuous batching schedule validated with CUDA token stepping\n";
  return 0;
}
