#include "../utils.hpp"
#include "layers.hpp"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iostream>
#include <vector_types.h>
template <typename T> class Layer {
private:
  int numInputChannels = 0;
  int numOutputChannels = 0;
  bool training = true; // change for training or validation passes.
  std::unique_ptr < Layer *next;
  Layer *prev;

public:
  // when forward is called from training loop, we will pass in the intermediate
  // data pointer, and have the same intermediate data pointer as d_output as
  // well. Logic here is that the same pointer is used to serve as the input and
  // output of a layer, similar to how a buffer would work.
  virtual void forward(T *d_input, T *d_output, int batch_size) = 0;
  // when backward is called from training loop, we will pass in the
  // intermediate data pointer for the gradient matrices (dE_dY), and then we
  // will also use dE_dX to pass to the previous layer
  virtual void backward(T *dE_dY, T *dE_dX, int batch_size) = 0;
  // returns the pointer to the output of a particular layer for its use in the
  // next layer.
  virtual void returnLayerPtr() = 0;
  Layer &connect(Layer &otherLayer) {
    this->next = &otherLayer;
    otherLayer.prev = this;

    return otherLayer;
  }
  // OPTIONAL: SAVING AND LOADING WEIGHTS
  // virtual void saveWeights() = 0;
  // virtual void loadWeights() = 0;
};

// Implementation of specific layers

template <typename T> class Linear : virtual public Layer<T> {
private:
  // weights (Input Nodes x Output Nodes) and biases (Input Nodes)
  T *weights;
  T *biases;

public:
  Linear(int numInputs, int numOutputs) {
    // Allocate device memory for inputs, weights, and biases
    CUDA_CHECK(
        cudaMalloc(&weights, batch_size * numOutputs * numInputs * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&biases, batch_size * numOutputs * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&outputs, batch_size * numOutputs * sizeof(T)));
    //  Initialize Weights with rand values
    int totalWeights = numInputs * numOutputs;
    int blockSize = 256;
    int gridSizeWeights = (totalWeights + blockSize - 1) / blockSize;

    curandState *weight_states;
    CUDA_CHECK(cudaMalloc(&weight_states, totalWeights * sizeof(curandState)));

    setup_curand_states<<<gridSizeWeights, blockSize>>>(weight_states,
                                                        time(NULL));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    initialize_values<<<gridSizeWeights, blockSize>>>(weights, weight_states,
                                                      totalWeights);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    // Initialize biases with zeroes
  }
  void forward(T *d_input, T *d_output, int batchSize) override {
    dim3 forwardBlockSize;
    dim3 forwardGridSize;

    linearForward<<<forwardGridSize, forwardBlockSize>>>();
  }
  void backward() override { std::cout << "TODO backward conv" << std::endl; }
  void update() { std::cout << "TODO update conv" << std::endl; }
  void returnLayerPtr() { return outputs; }
};

template <typename T> class Conv : virtual public Layer<T> {
private:
  T *kernel;
  int numInputsChannels;
  int numOutputsChannels;
  int kernelWidth;
  int kernelHeight;
  int totalElements;
  int stride;
  int padding;

public:
  Conv(int numInputs, int numOutputs, int kernelSize, int stride, int padding)
      : numInputChannels(numInputs), numOutputChannels(numOutputs),
        kernelHeight(kernelSize), kernelWidth(kernelSize), stride(stride),
        padding(padding), totalElements(kernelSize * kernelSize *
                                        numInputChannels * numOutputsChannels) {
    CUDA_CHECK(cudaMalloc(&kernel, sizeof(T) * totalElements));
    CUDA_CHECK(cudaMalloc(&outputs, sizeof(T) * numOutputs));
    curandState *kernel_state;
    CUDA_CHECK(cudaMalloc(&kernel_state, sizeof(curandState) * totalElements));
    // Define block and grid dimensions for kernel launch
    int threadsPerBlock = 256; // Common choice, can be 512 as well
    dim3 kernelBlockDim(threadsPerBlock);

    // Calculate the number of blocks needed
    dim3 kernelGridDim((totalElements + threadsPerBlock - 1) / threadsPerBlock);
    setup_curand_states<<<kernelGridDim, kernelBlockDim>>>(
        &kernel_state, totalElements * sizeof(curandState));
    CUDA_CHECK(cudaGetLastError());

    initialize_values<<<kernelGridDim, stateBlockDim>>>(&kernel, kernel_state,
                                                        totalElements);
    CUDA_CHECK(cudaGetLastError());
  }

  ~Conv() {}

  void forward(T *d_input, T *d_output, int batchSize) override {
    std::cout << "TODO forward conv" << std::endl;
  }
  void backward() override { std::cout << "TODO backward conv" << std::endl; }
  void update() { std::cout << "TODO update conv" << std::endl; }
  void returnLayerPtr() override { return outputs; }
};

//
// template <typename T> class Flatten : virtual public Layer<T> {
//
// private:
//  int startDim;
//
// public:
//  Flatten() : startDim(startDim) {}
//  ~Flatten() {}
//  void forward() override { std::cout << "TODO forward flatten" << std::endl;

//    std::cout << "TODO backward flatten" << std::endl;
//  }
//  void update() { std::cout << "TODO update flatten" << std::endl; }
//};
//
// TODO later
// template <typename T> class BatchNorm : virtual public Layer<T> {
//
// private:
//   int numInputs;
//
// public:
//   BatchNorm(int numInputs) : numInputs(numInputs) {
//
// }
//
//   void forward() override {
//     std::cout << "TODO forward batch norm" << std::endl;
//   }
//
//  void backward() override {
//     std::cout << "TODO backward batch norm" << std::endl;
//   }
//   void update() { std::cout << "TODO update batch norm" << std::endl; }
// };

template <typename T> class Pooling : virtual public Layer<T> {
private:
  int kernelSize;
  int stride;

public:
  Pooling(int kernelSize, int stride) : kernelSize(kernelSize), stride(stride) {
    CUDA_CHECK(cudaMalloc(&outputs, sizeof(T) * FIGURE_OUT_LATER));
  }
  ~Pooling() {}

  void forward(T *d_input, T *d_output, int batchSize) override {
    std::cout << "TODO forward pooling" << std::endl;
  }
  void backward() override {
    std::cout << "TODO backward pooling" << std::endl;
  }
  void update() { std::cout << "TODO update pooling" << std::endl; }
  void returnLayerPtr() override { return outputs; }
};

template <typename T> class ReLu : virtual public Layer<T> {

public:
  ReLu(T *inputPtr, T *outputPtr, int numInputs, int batchSize)
      : Layer<T>(numInputs) {
    std::cout << "TODO ReLU constructor" << std::endl;
    // allocate device array if you need to keep it
    // CUDA_CHECK(cudaMalloc(&d_input, sizeof(T) * this->_numInputs));
  }
  //~ReLu() { CUDA_CHECK(cudaFree(d_input)); }
  void forward(T *d_input, T *d_output, int size) override {
    // launch your CUDA kernel here, e.g., relu_forward_kernel<<<...>>>(...)
    std::cout << "Running ReLU forward on " << size << " elements" << std::endl;
  }
  void backward() override { std::cout << "TODO backward RELU" << std::endl; }
};

template <typename T> class Softmax : virtual public Layer<T> {

private:
  int numInputs;

public:
  Softmax(T *d_input, int numInputs) : numInputs(numInputs) {}
  ~Softmax() {}
  void forward() override { std::cout << "TODO forward Softmax" << std::endl; }
  void backward() override {
    std::cout << "TODO backward Softmax" << std::endl;
  }
  void update() { std::cout << "TODO update Softmax" << std::endl; }
};
