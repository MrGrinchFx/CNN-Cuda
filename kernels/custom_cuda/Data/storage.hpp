#pragma once

#include <cuda_runtime.h>
#include <iterator>
#include <thrust/device_vector.h>
#include <vector>

template <typename T> class Container {
public:
  explicit Container(const std::vector<int> &_shape);
  explicit Container(const std::vector<int> &_shape, T value);
  explicit Container(const std::vector<int> &_shape,
                     const std::vector<T> &_data);
  // move and copy operations/constructors
  Container(const Container &other);
  Container(const Container &&other);
  Container &operator=(const Container &other);
  Container &operator=(const Container &&other);

  void reshape(const std::vector<int> _shape);
  void resize(const std::vector<int> _shape);
  void xavierInit(size_t inSize, size_t outSize);

  // get functions

  std::vector<int> &getShape() { return this->shape; }
  const std::vector<int> &getShape() const { return this->shape; }
  const thrust::device_vector<T> &getData() const { return this->data; }
  thrust::device_vector<T> &getData() { return this->data; }

private:
  void checkSize();
  std::vector<int> shape;
  thrust::device_vector<T> data;
};
