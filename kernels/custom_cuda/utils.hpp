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

// data loader
std::vector<u_int8_t> load_mnist_images(const std::string &filename,
                                        int num_images, int image_size);
std::vector<u_int8_t> load_mnist_labels(const std::string &filename,
                                        int num_labels);
