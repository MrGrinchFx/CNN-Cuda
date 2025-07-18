#include "utils.hpp"
#include <cuda_device_runtime_api.h>
#include <curand_kernel.h>

__global__ void setup_curand_states(curandState *states, unsigned long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(seed, idx, 0, &states[idx]);
}

__global__ void initialize_values(float *array, curandState *states, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    curandState local = states[idx];
    array[idx] = curand_normal(&local) * 0.01f;
    states[idx] = local;
  }
}
