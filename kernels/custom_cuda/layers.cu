#pragma once
#include "layerKernels.cu"
#include "utils.hpp"
#include <algorithm>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <curand_kernel.h>
#include <iostream>
#include <random>
#include <vector_types.h>
template <typename T> class Layer {
protected:
  int numInputs = 0;
  int numOutputs = 0;
  bool training = true; // change for training or validation passes.

public:
  virtual void forward(T *d_input, T *d_output, int batch_size) = 0;
  virtual void backward(T *d_output_grad, int batch_size) = 0;

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
    // Allocate device memory for inputs, outputs, weights, and biases
    CUDA_CHECK(cudaMalloc(&weights, numOutputs * numInputs * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&biases, numOutputs * sizeof(T)));

    // === Initialize Weights ===
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

    // === Initialize Biases ===
    int totalBiases = numOutputs;
    int gridSizeBiases = (totalBiases + blockSize - 1) / blockSize;

    curandState *bias_states;
    CUDA_CHECK(cudaMalloc(&bias_states, totalBiases * sizeof(curandState)));

    setup_curand_states<<<gridSizeBiases, blockSize>>>(
        bias_states, time(NULL) + 1234); // Different seed
    CUDA_CHECK(cudaGetLastError());

    initialize_values<<<gridSizeBiases, blockSize>>>(biases, bias_states,
                                                     totalBiases);
    CUDA_CHECK(cudaGetLastError());

    // Optional: free temporary RNG state memory if you won’t reuse it
    CUDA_CHECK(cudaFree(weight_states));
    CUDA_CHECK(cudaFree(bias_states));
  }
  void forward(T *d_input, T *d_output) override {}
  void backward() override { std::cout << "TODO backward conv" << std::endl; }
  void update() { std::cout << "TODO update conv" << std::endl; }
};

template <typename T> class Conv : virtual public Layer<T> {
private:
  T *kernel;
  int numInputs_;
  int numOutputs_;
  int kernelSize_;
  int totalElements;
  int stride_;
  int padding_;

public:
  Conv(int numInputs, int numOutputs, int kernelSize, int stride, int padding)
      : numInputs_(numInputs), numOutputs_(numOutputs), kernelSize_(kernelSize),
        stride_(stride), padding_(padding),
        totalElements(kernelSize * kernelSize) {
    CUDA_CHECK(cudaMalloc(&kernel, sizeof(T) * totalElements));

    curandState *kernel_state;
    CUDA_CHECK(cudaMalloc(&kernel_state, sizeof(curandState) * totalElements));
    // Define block and grid dimensions for kernel launch
    int threadsPerBlock = 256; // Common choice, can be 512 as well
    dim3 kernelBlockDim(threadsPerBlock);

    // Calculate the number of blocks needed
    dim3 kernelGridDim((totalElements + threadsPerBlock - 1) / threadsPerBlock);
    setup_curand_states<<<kernelGridDim, kernelBlockDim>>>(&kernel_state,
                                                           totalElements * sizeof(curandState);
    CUDA_CHECK(cudaGetLastError());

    initialize_values<<<stateGridDim, stateBlockDim>>>(&kernel, kernel_state,
                                                       totalElements);
    CUDA_CHECK(cudaGetLastError());
  }

  ~Conv() {}

  void forward(T *d_input, T *d_output) override {
    std::cout << "TODO forward conv" << std::endl;
  }
  void backward() override { std::cout << "TODO backward conv" << std::endl; }
  void update() { std::cout << "TODO update conv" << std::endl; }
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
//  } void backward() override {
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
  Pooling(int kernelSize, int stride)
      : kernelSize(kernelSize), stride(stride) {}
  ~Pooling() {}

  void forward(T *d_input, T *d_output) override {
    std::cout << "TODO forward pooling" << std::endl;
  }
  void backward() override {
    std::cout << "TODO backward pooling" << std::endl;
  }
  void update() { std::cout << "TODO update pooling" << std::endl; }
};

template <typename T> class ReLu : virtual public Layer<T> {

public:
  ReLu(int numInputs) : Layer<T>(numInputs) {
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
