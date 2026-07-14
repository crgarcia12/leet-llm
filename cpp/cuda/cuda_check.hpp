#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

inline void cuda_check(cudaError_t status, const char* expression, const char* file, int line) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expression,
                 cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
  }
}
#define CUDA_CHECK(expression) cuda_check((expression), #expression, __FILE__, __LINE__)
