#include "Layers/layers.hpp"
#include <memory>
#include <vector>
template <typename T> class Architecture {
private:
  std::vector<std::unique_ptr<Layer<T>>> layersVector;
  // stores upstream inputs
  std::vector<std::unique_ptr<T>> fwdIntermediatePtrs;
  // stores upstream gradients
  std::vector<std::unique_ptr<T>> bkwdIntermediatePtrs;
  int numClasses;
  int imgHeight;
  int imgWidth;
  int batchSize;

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
      for (int fwd = 0; fwd < layersVector.size(); fwd++) {
        // TODO forward propogation logic
        // Go through each layer and pass into the input
        // fwdIntermediatePtrs[fwd] for the corresponding output of the previous
        // layer
      }
      // TODO perform SOFTMAX AND LOSS CALC
      for (int bkwd = layersVector.size() - 1; bkwd > 0; bkwd--) {
        // TODO backpropogation logic
        // Go throuh each layer backwards and pass into the input
        // bkwdIntermediatePtrs[bkwd] for the corresponding gradient of the next
        // layer into the current layer
      }
      // print results of the training (i.e accuracy, training time, and etc.)
      printResults();
    }
  }
  void setNumClasses(int classesAmt) { numClasses = classesAmt; }
  void setInputDim(int height, int width) {
    imgHeight = height;
    imgWidth = width;
  }
  void addBatchSize(int batchSize_) { batchSize = batchSize_; }
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
    layersVector.push_back(std::make_unique<ReLu<T>>(numInputs));
  }

  void addMaxPool(int kernelSize, int stride) {
    layersVector.push_back(std::make_unique<Pooling<T>>(kernelSize, stride));
  }

  // void addBatchNorm(int numFeatures) {
  //   layersVector.push_back(std::make_unique<BatchNorm<T>>(numFeatures));
  // }

  // void addFlatten(int startDim) {
  //   layersVector.push_back(std::make_unique<Flatten<T>>(startDim));
  // }
};
