#pragma once

#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " ";     \
      std::cerr << "code=" << static_cast<int>(err) << " ("                    \
                << cudaGetErrorString(err) << ")" << std::endl;                \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)
