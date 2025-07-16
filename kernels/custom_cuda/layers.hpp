#pragma once

#include "utils.hpp"
#include <algorithm>
#include <iostream>
#include <random>
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
  std::vector<T> weights_;
  std::vector<T> biases_;
  std::vector<T> input_d;
  std::vector<T> output_d;

public:
  Linear(int numInputs, int numOutputs) {
    weights_.resize(numInputs * numOutputs);
    biases_.resize(numOutputs);
    input_d.resize(numInputs);
    output_d.resize(numOutputs);

    // generate random weights and biases
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<T> dist(static_cast<T>(-10.0),
                                           static_cast<T>(10.0));
    std::generate(weights_.begin(), weights_.end(),
                  [&]() { return dist(rng); });
    std::generate(biases_.begin(), biases_.end(), [&]() { return dist(rng); });
  }
  void forward() override { std::cout << "TODO forward conv" << std::endl; }
  void backward() override { std::cout << "TODO backward conv" << std::endl; }
  void update() { std::cout << "TODO update conv" << std::endl; }
};

template <typename T> class Conv : virtual public Layer<T> {
private:
  std::vector<T> kernel;
  std::vector<T> input_d;
  std::vector<T> output_d;
  int numInputs_;
  int numOutputs_;
  int kernelSize_;
  int stride_;
  int padding_;

public:
  Conv(int numInputs, int numOutputs, int kernelSize, int stride, int padding)
      : numInputs_(numInputs), numOutputs_(numOutputs), kernelSize_(kernelSize),
        stride_(stride), padding_(padding) {
    kernel.resize(kernelSize * kernelSize);
  }

  ~Conv() {}

  void forward() override { std::cout << "TODO forward conv" << std::endl; }
  void backward() override { std::cout << "TODO backward conv" << std::endl; }
  void update() { std::cout << "TODO update conv" << std::endl; }
};

template <typename T> class Flatten : virtual public Layer<T> {

private:
  int startDim;

public:
  Flatten() : startDim(startDim) {}
  ~Flatten() {}
  void forward() override { std::cout << "TODO forward flatten" << std::endl; }
  void backward() override {
    std::cout << "TODO backward flatten" << std::endl;
  }
  void update() { std::cout << "TODO update flatten" << std::endl; }
};

template <typename T> class BatchNorm : virtual public Layer<T> {

private:
  int numInputs;

public:
  BatchNorm(int numInputs) : numInputs(numInputs) {}
  void forward() override {
    std::cout << "TODO forward batch norm" << std::endl;
  }
  void backward() override {
    std::cout << "TODO backward batch norm" << std::endl;
  }
  void update() { std::cout << "TODO update batch norm" << std::endl; }
};

template <typename T> class Pooling : virtual public Layer<T> {
private:
  int kernelSize;
  int stride;

public:
  Pooling(int kernelSize, int stride)
      : kernelSize(kernelSize), stride(stride) {}
  ~Pooling() {}

  void forward() override { std::cout << "TODO forward pooling" << std::endl; }
  void backward() override {
    std::cout << "TODO backward pooling" << std::endl;
  }
  void update() { std::cout << "TODO update pooling" << std::endl; }
};

template <typename T> class ReLu : virtual public Layer<T> {

private:
  T *d_input = nullptr; // optional, depends on your design

public:
  ReLu(int numInputs) : Layer<T>(numInputs) {
    std::cout << "TODO ReLU constructor" << std::endl;
    // allocate device array if you need to keep it
    // CUDA_CHECK(cudaMalloc(&d_input, sizeof(T) * this->_numInputs));
  }
  //~ReLu() { CUDA_CHECK(cudaFree(d_input)); }
  void forward(T *d_input, int size) override {
    // launch your CUDA kernel here, e.g., relu_forward_kernel<<<...>>>(...)
    std::cout << "Running ReLU forward on " << size << " elements" << std::endl;
  }
  void backward() override { std::cout << "TODO backward RELU" << std::endl; }
};

template <typename T> class Softmax : virtual public Layer<T> {

private:
  int numInputs;

public:
  Softmax(int numInputs) : numInputs(numInputs) {}
  ~Softmax() {}
  void forward() override { std::cout << "TODO forward Softmax" << std::endl; }
  void backward() override {
    std::cout << "TODO backward Softmax" << std::endl;
  }
  void update() { std::cout << "TODO update Softmax" << std::endl; }
};
