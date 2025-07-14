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
  void train() {
    // TODO
    // We're going to have 2 inner loops for both Forward and Backward
    // propagation, as well as the calling of the softmax and loss function
    // computation in between both loops. This will be wrapped around a larger
    // for loop that will load batches of data into the model each iteration. We
    // will also handle the data loading in here most likely."addMaxPool"
  }

  void addConv(int numInputs, int numOutputs, int kernelSize, int stride,
               int padding) {
    Conv<T> *layer =
        new Conv<T>(numInputs, numOutputs, kernelSize, stride, padding);
    layersVector.push_back(std::make_unique(layer));
  }

  void printResults() {
    // TODO
  }
  void addLinear(int numInputs, int numOutputs) {
    // Responsible for taking in tuple arguments
    Linear<T> *layer = new Linear<T>(numInputs, numOutputs);
    layersVector.push_back(layer);
  }

  void addRelu(int numInputs) {
    ReLu<T> *layer = new ReLu<T>(numInputs);
    layersVector.push_back(std::make_unique(layer));
  }

  void addMaxPool(int kernelSize, int stride) {
    Pooling<T> *layer = new Pooling<T>(kernelSize, stride);
    layersVector.push_back(std::make_unique(layer));
  }

  void addBatchNorm(int numFeatures) {
    BatchNorm<T> *layer = new BatchNorm<T>(numFeatures);
    layersVector.push_back(std::make_unique(layer));
  }

  void addFlatten(int startDim) {
    Flatten<T> *layer = new Flatten<T>(startDim);
    layersVector.push_back(std::make_unique(layer));
  }
};
