#include "layers.hpp"
#include <iostream>
#include <memory>
#include <vector>
template <typename T> class Architecture {
private:
  std::vector<std::unique_ptr<Layer<T>>> layersVector;

public:
  Architecture<T>() {}
  std::vector<Layer<T>> getLayersVector() { return layersVector; }
  void train(int trainIterations) {
    // TODO
    // We're going to have 2 inner loops for both Forward and Backward
    // propagation, as well as the calling of the softmax and loss function
    // computation in between both loops. This will be wrapped around a larger
    // for loop that will load batches of data into the model each iteration. We
    // will also handle the data loading in here most likely."addMaxPool"

    // main loops
    for (int i = 0; i < trainIterations; i++) {
      for (auto it = layersVector.begin(); it != layersVector.end(); i++) {
        // TODO forward propogation logic
      }
      // TODO perform SOFTMAX AND LOSS CALC
      for (auto it = layersVector.rbegin(); it != layersVector.rend(); it++) {
        // TODO backpropogation logic
      }
      // print results of the training (i.e accuracy, training time, and etc.)
      printResults();
    }
  }

  void addConv(int numInputs, int numOutputs, int kernelSize, int stride,
               int padding) {
    layersVector.push_back(std::make_unique<Conv<T>>(
        numInputs, numOutputs, kernelSize, stride, padding));
  }

  void printResults() {
    // TODO
  }
  void addLinear(int numInputs, int numOutputs) {
    // Responsible for taking in tuple arguments
    layersVector.push_back(std::make_unique<Linear<T>>(numInputs, numOutputs));
  }

  void addRelu(int numInputs) {
    layersVector.push_back(std::make_unique<ReLu<T>>(numInputs));
  }

  void addMaxPool(int kernelSize, int stride) {
    layersVector.push_back(std::make_unique<Pooling<T>>(kernelSize, stride));
  }

  void addBatchNorm(int numFeatures) {
    layersVector.push_back(std::make_unique<BatchNorm<T>>(numFeatures));
  }

  void addFlatten(int startDim) {
    layersVector.push_back(std::make_unique<Flatten<T>>(startDim));
  }
};
