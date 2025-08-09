#include "Data/dataloader.cuh"
#include "Layers/conv.cuh"
#include "Layers/layers.cuh"
#include "Layers/linear.cuh"
#include "Layers/maxPool.cuh"
#include "Layers/relu.cuh"
#include "Optimizer/sgd-std.cuh"
#include <memory>
#include <vector>
template <typename T> class Architecture {
private:
  std::vector<std::unique_ptr<Layer<T>>> layersVector;
  std::unique_ptr<Optimizer<T>> optimizer;

public:
  Architecture<T>(std::unique_ptr<Dataloader<T>> &&dataloader) {
    // Dataloader is the first "layer"
    layersVector.push_back(std::move(dataloader));
  }

  std::vector<Layer<T>> getLayersVector() { return layersVector; }
  void train(int trainIterations, float learningRate, float l2) {
    // connect the layers via pointers
    connectLayers();
    // setup the optimizer with the parameters it needs to tune

    auto allParams = getAllParameters();
    optimizer.reset(new SGD<T>(learningRate, l2));
    optimizer->reg(allParams);

    while (1) { // main loops
      for (int fwd = 0; fwd < layersVector.size(); fwd++) {
        // TODO
      }
      for (int bkwd = layersVector.size() - 1; bkwd > 0; bkwd--) {
        // TODO
      }
    }
    printResults();
  }

  void connectLayers() {
    for (auto it = layersVector.begin(); it != layersVector.end(); it++) {
      // check if there is a next layer
      if (std::next(it) != layersVector.end()) {
        (*it)->connect(**std::next(it));
      }
    }
  }

  std::vector<std::pair<Container<T> *, Container<T> *>> getAllParameters() {
    std::vector<std::pair<Container<T> *, Container<T> *>> allParams;
    for (const auto &layer : layersVector) {
      auto params = layer->parameters(); // a method each layer has
      if (!params.empty()) {
        allParams.insert(allParams.end(), params.begin(), params.end());
      }
    }
    return allParams;
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
    layersVector.push_back(std::make_unique<Linear<T>>(numInputs, numOutputs));
  }

  void addRelu(int numInputs) {
    layersVector.push_back(std::make_unique<Relu<T>>(numInputs));
  }

  void addMaxPool(int kernelSize, int stride) {
    layersVector.push_back(std::make_unique<Maxpool<T>>(kernelSize, stride));
  }

  // void addBatchNorm(int numFeatures) {
  //   layersVector.push_back(std::make_unique<BatchNorm<T>>(numFeatures));
  // }

  void addFlatten(int startDim) {
    layersVector.push_back(std::make_unique<Flatten<T>>(startDim));
  }
};
