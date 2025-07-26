#include "cuda_runtime_api.h"
#include "utils.hpp"
#include <iostream>
#include <neuralnet.cpp>

int main() {
  // Starting point of the include
  // Populate the Weights and biases with random numbers (Can be of any type)
  // Call the constructor for an architecture object (it will accept a vector of
  // layers and activations) Call a MLP.train() that will conduct training (Will
  // define training logic in a separate file)
  auto cudaStatus = cudaSetDevice(0);
  CHECK_EQ(cudaStatus, cudaSuccess,
           "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  Architecture<float> myModel;
  int numClasses = 10; // number of different digits in MNIST

  load_mnist_images("../../data/MNIST/raw/train-images-idx3-ubyte", 60000, 784);
  load_mnist_labels("../../data/MNIST/raw/train-labels-idx1-ubyte", 60000);
  load_mnist_images("../../data/MNIST/raw/t10k-images-idx3-ubyte", 10000, 784);
  load_mnist_labels("../../data/MNIST/raw/t10k-labels-idx1-ubyte", 10000);
  // example usage | check ~/python/python_reference.ipynb for the pytorch
  // equivalent of this model!
  // First call for the creation of a Dataset and then adding layers will
  // involve "connecting" them
  myModel.setNumClasses(numClasses);
  myModel.setInputDim(28, 28);
  myModel.addBatchSize(64);
  myModel.addConv(1, 32, 5, 2, 1);
  myModel.addMaxPool(2, 2);
  myModel.addConv(32, 64, 5, 1, 2);
  myModel.addLinear(14 * 14 * 64, 1024);
  myModel.addRelu(1024);
  myModel.addLinear(1024, numClasses);
  myModel.train(100);
  myModel.printResults();

  return 0;
}
