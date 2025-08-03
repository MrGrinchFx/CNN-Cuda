#include "../Data/container.cuh"
#include <curand.h>
#include <curand_kernel.h>

template <typename T>
void operatorMatMul(Container<T> *input1, Container<T> *input2,
                    Container<T> *output, int broadcast);

template <typename T>
void __global__ matMulKernel(const T *input1, const T *input2, T *output,
                             int height, int width, int k, int broadcast);

template <typename T>
void __global__ xavierKernel(T *a, int size, float scale, curandState *cs);
