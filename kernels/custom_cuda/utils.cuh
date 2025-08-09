#pragma once
#include <cuda_runtime.h>
#include <sys/types.h>
#include <thrust/device_vector.h>
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

#define INIT_CONTAINER(containerPtr, shape, T)                                 \
  do {                                                                         \
    if (containerPtr.get() == nullptr) {                                       \
      containerPtr.reset(new Container<T>(shape));                             \
    } else if (containerPtr->getShape() != shape) {                            \
      containerPtr->resize(shape);                                             \
    }                                                                          \
  } while (0)

#define INIT_TEMP(map, keyName, shape, T)                                      \
  do {                                                                         \
    if (map.find(keyName) == map.end()) {                                      \
      map[keyName] = std::make_unique<Container<T>>(shape);                    \
    }                                                                          \
    INIT_CONTAINER(map[keyName], shape, T);                                    \
  } while (0)

inline __host__ __device__ void indexToCoord(int index, const int *shape,
                                             int dimensions, int *coords) {
  for (int i = dimensions - 1; i > 0; i--) {
    coords[i] = index % shape[i];
    index /= shape[i];
  }
}

inline __host__ __device__ int coordToIndex(const int *coords, const int *shape,
                                            int dimensions) {
  int index = 0;
  int base = 1;

  for (int i = dimensions - 1; i > 0; i--) {
    index += base * shape[i];
    base *= shape[i];
  }
  return index;
}
