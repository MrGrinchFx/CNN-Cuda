#pragma once
#include <cstdint>  // uint8_t, uint32_t
#include <fstream>  // ifstream, file I/O
#include <iostream> // cout, cerr
#include <memory>   // std::unique_ptr, std::make_unique
#include <stdexcept>
#include <string> // std::string
#include <sys/types.h>
#include <tuple>  // std::tuple if you use it
#include <vector> // std::vector
#define BLOCK_SIZE 256
#define TILE_SIZE 16

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

#define CHECK_EQ(val1, val2, message)                                          \
  do {                                                                         \
    if (val1 != val2) {                                                        \
      std::cerr << __FILE__ << "(" << __LINE__ << "): " << message             \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define RAW_PTR(vector) thrust::raw_pointer_cast(vector.data())
