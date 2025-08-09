#include "Data/dataloader.cuh"
#include "cuda_runtime_api.h"
#include "neuralnet.cu"
#include "utils.cuh"
#include <iostream>
int main() {
  // Starting point of the include
  // Populate the Weights and biases with random numbers (Can be of any type)
  // Call the constructor for an architecture object (it will accept a vector of
  // layers and activations) Call a MLP.train() that will conduct training (Will
  // define training logic in a separate file)
  auto cudaStatus = cudaSetDevice(0);
  CHECK_EQ(cudaStatus, cudaSuccess,
           "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
  auto dataloader =
      std::make_unique<Dataloader<float>>("../../data/MNIST/raw/");
  auto myModel = new Architecture<float>(std::move(dataloader));
  // example usage | check ~/python/python_reference.ipynb for the pytorch
  // equivalent of this model!
  // First call for the creation of a Dataset and then adding layers will
  // involve "connecting" them
  myModel->addConv(1, 32, 5, 2, 1);
  myModel->addMaxPool(2, 2);
  myModel->addConv(32, 64, 5, 1, 2);
  myModel->addLinear(14 * 14 * 64, 1024);
  myModel->addRelu(1024);
  myModel->addLinear(1024, 10);
  myModel->train();
  myModel->printResults();

  return 0;
}
