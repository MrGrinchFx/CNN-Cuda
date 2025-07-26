#include "../utils.hpp"
#include "curand_kernel.h"
#include "dataloader.hpp"
#include "storage.hpp"
#include <cmath>
#include <curand.h>
#include <device_launch_parameters.h>
#include <vector>

template <typename T>
Container<T>::Container(const std::vector<int> &_shape) : shape(_shape) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  return this->data.resize(size);
}

template <typename T>
Container<T>::Container(const std::vector<int> &_shape, T value)
    : shape(_shape) {
  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }
  return this->data.resize(size, value);
}
//
template <typename T>
Container<T>::Container(const std::vector<int> &_shape,
                        const std::vector<T> &_data)
    : shape(_shape), data(_data.begin(), _data.end()) {
  this->checkSize();
}
// copy constructor
template <typename T> Container<T>::Container(const Container &other) {
  // call the move assignment
  *this = other;
}
template <typename T>
Container<T> &Container<T>::operator=(const Container<T> &other) {
  if (this != &other) {
    // ownership is exchanged
    this->shape = std::move(other.shape);
    this->data = std::move(other.data);
  }

  return *this;
}
template <typename T>
void Container<T>::reshape(const std::vector<int> _shape) {
  this->shape = _shape;
  this->checkSize();
}
template <typename T> void Container<T>::resize(const std::vector<int> _shape) {
  this->shape = _shape;

  int size = 1;
  for (int i = 0; i < _shape.size(); i++) {
    size *= _shape[i];
  }

  if (size != this->data.size()) {
    this->data.resize(size);
  }
}

__global__ void xavierKernel(float *a, int size, float scale, curandState *cs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curand_init(1234, index, 0, &cs[index]);
    a[index] = (curand_uniform(&cs[index]) * 2 - 1) * scale;
  }
}

template <typename T>
void Container<T>::xavierInit(size_t inSize, size_t outSize) {
  // CUDA doesn't take in smart pointers, so we have to strip the ptrs down to a
  // raw pointer since the Container object stores it as a unique_ptr rather
  // than a raw one.
  T *ptr = RAW_PTR(this->data);
  int size = this->data.size();
  int gridSize = ceil((T)size, BLOCK_SIZE);

  thrust::device_vector<curandState> states(size);
  curandState *statePtr = RAW_PTR(states);
  T scale = std::sqrt((T)6) / std::sqrt((T)(inSize) + outSize);
  xavierKernel<<<gridSize, BLOCK_SIZE>>>(ptr, size, scale, statePtr);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T> void Container<T>::checkSize() {
  int size = 1;

  for (int i = 0; i < this->shape.size(); i++) {
    size *= this->shape[i];
  }
  CHECK_EQ(size, this->data.size(), "Container sizes dont match");
}
