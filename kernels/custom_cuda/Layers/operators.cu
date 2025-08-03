#include "operators.cuh"

template <typename T>
void __global__ matMulKernel(const T *a, const T *b, T *c, int height,
                             int width, int k, int broadcast) {
  // TODO
}

template <typename T>
void operatorMatMul(Container<T> *a, Container<T> *b, Container<T> *c,
                    int broadcast) {
  // TODO
}

template <typename T>
void __global__ xavierKernel(T *a, int size, float scale, curandState *cs) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    curand_init(1234, index, 0, &cs[index]);
    a[index] = (curand_uniform(&cs[index]) * 2 - 1) * scale;
  }
}
