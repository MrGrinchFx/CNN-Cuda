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

template <typename T>
__global__ void operatorSumKernel(const T *inputPtr, T *outputPtr,
                                  const int *inputShapePtr, int *tempShapePtr,
                                  int inputDims, int dim, int dimStride,
                                  int size) {
  // extern to tell the compiler that sharedMemSize will be determined at
  // runtime.
  extern __shared__ int shared[];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int *coord = (int *)shared + threadIdx.x * inputDims;

    // convert from flattened index to coordinates.
    indexToCoord(index, tempShapePtr, inputDims - 1, coord);

    for (int i = inputDims - 1; i > dim; i--) {
      coord[i] = coord[i - 1];
    }

    coord[dim] = 0;
    int base = coordToIndex(coord, inputShapePtr, inputDims);
    // go through target dimension and reduce
    int length = inputShapePtr[dim];
    double total = 0;
    for (int i = 0; i < length; i++) {
      total += inputPtr[base + i * dimStride];
      outputPtr[index] = total;
    }
  }
}

template <typename T>
void operatorSum(const Container<T> *input, int dim,
                 const Container<T> *output) {
  // Parameters: input vector, dimension in which reduction is performed, and
  // output vector
  const T *inputPtr = thrust::raw_pointer_cast(input);
  thrust::device_vector<T> inputShape = input->getShape();
  int inputDims = input->getShape().size();
  const int *inputShapePtr = thrust::raw_pointer_cast(inputShape);

  const T *outputPtr = thrust::raw_pointer_cast(output);
  thrust::device_vector<T> tempShape = inputShape;
  // sum reduction of a dimension results in the loss of that chosen dimension
  tempShape.erase(tempShape.begin() + dim);
  const T *tempShapePtr = thrust::raw_pointer_cast(tempShape);

  int dimStride = 1;
  for (int i = dim + 1; i < inputDims; i++) {
    dimStride *= inputShape->getShape()[i];
  }

  //(total data points / total data points in target dimension)
  int size = input->getData().size() / input->getShape()[dim];
  int gridSize = ceil((float)(size) / BLOCK_SIZE);
  int sharedMemSize = BLOCK_SIZE * inputDims * sizeof(int);

  operatorSumKernel<<<gridSize, BLOCK_SIZE, sharedMemSize>>>(
      inputPtr, outputPtr, inputShapePtr, tempShapePtr, inputDims, dim,
      dimStride, size);

  CUDA_CHECK(cudaGetLastError());
}
