#include "../Data/container.cuh"
#include "layers.cuh"
#include <memory>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
template <typename T>
void linearFwdOp(const Container<T> *inputs, const Container<T> *weights,
                 Container<T> *outputs);

template <typename T>
void linearBwdOp(
    const Container<T> *inputs, const Container<T> *weights,
    const Container<T> *outputGrad, Container<T> *weightsGrad,
    Container<T> *inputsGrad,
    std::unordered_map<std::string, std::unique_ptr<Container<T>>> &temp);

template <typename T>
void linearFwdBiasOp(const Container<T> *inputs, const Container<T> *biases,
                     const Container<T> *outputs);

template <typename T>
void linearBwdBiasOp(const Container<T> *outputGrad, Container<T> *biasesGrad);

template <typename T> class Linear : Layer<T> {
public:
  explicit Linear(int inputChannels, int outputChannels);
  void forward();
  void backward();

  std::vector<std::pair<Container<T> *, Container<T> *>>
  getParameters(); // returns the weight and weightGrads as well as bias and
                   // biasGrads

private:
  std::unique_ptr<Container<T>> weights;
  std::unique_ptr<Container<T>> weightGrads;
  std::unique_ptr<Container<T>> bias;
  std::unique_ptr<Container<T>> biasGrads;

  std::unordered_map<std::string, std::unique_ptr<Container<T>>> temp;

  int inputChannels;
  int outputChannels;
};
