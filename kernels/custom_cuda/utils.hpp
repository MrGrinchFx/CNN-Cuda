#pragma once
#include <sys/types.h>
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

#define INIT_CONTAINER(container_ptr, shape)                                   \
  do {                                                                         \
    if (container_ptr.get() == nullptr) {                                      \
      container_ptr.reset(new Container<T>(shape));                            \
    } else if (container_ptr->getShape() != shape) {                           \
      container_ptr->resize(shape);                                            \
    }                                                                          \
  } while (0)
