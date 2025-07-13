#include "layers.hpp"
#include <iostream>
#include <vector>

template <typename T> class Architecture {
private:
  std::vector<Layer<T>> layersVector;

public:
  Architecture<T>() {}
  std::vector<Layer<T>> getLayersVector() { return layersVector; }
  void addLinear(int numInputs, int numOutputs) {
    // Responsible for taking in tuple arguments
  }
  void train() {
    // TODO
  }
  void addConv(int numInputs, int numOutputs, int kernelSize, int stride,
               int padding) {
    // TODO
  }

  void addRelu() {
    // TODO
  }

  void addMaxPool(int kernelSize, int stride) {
    // TODO
  }

  void addFlatten(int startDim) {
    // TODO
  }
};
