
#include "linear.cuh"
#include "operators.cuh"
#include <thrust/fill.h>

template <typename T> Linear<T>::Linear(int inputChannels, int outputChannels) {

  this->inputChannels = inputChannels;
  this->outputChannels = outputChannels;
  // weights initialization
  this->weights.reset(new Container<T>({inputChannels, outputChannels}));
  this->weightGrads.reset(new Container<T>({inputChannels, outputChannels}));
  this->weights->xavierInit(inputChannels,
                            outputChannels); // random initialization
  // bias initialization
  this->bias.reset(new Container<T>({1, outputChannels}));
  this->biasGrads.reset(new Container<T>({1, outputChannels}));
  thrust::fill(this->bias->getData().begin(), this->bias->getData().end(),
               0); // initialze all to zero rather than call xavier
                   // kernel
}

template <typename T>
std::vector<std::pair<Container<T> *, Container<T> *>>
Linear<T>::getParameters() {
  return {std::make_pair(this->weights.get(), this->weightGrads.get()),
          std::make_pair(this->bias.get(), this->biasGrads.get())};
}

template <typename T> void Linear<T>::forward() {
  // prev layer's output becomes current layer's input
  const Container<T> *input = this->prev->getOutput();
  std::vector<int> outputShape = {
      input->getShape()[0],
      this->outputChannels}; //{batchSize, outputChannels};

  INIT_CONTAINER(this->output, outputShape);

  linearFwdOp(input, this->weights.get(), this->output.get());
  linearFwdBiasOp(input, this->biases.get(), this->output.get());
}

template <typename T> void Linear<T>::backward() {
  const Container<T> *input = this->prev->getOutput();
  const Container<T> *outputGrad = this->next->getGrad();

  INIT_CONTAINER(this->grad, input->getShape());

  linearBwdBiasOp(outputGrad, this->biasGrads.get());

  linearBwdOp(input, this->weights.get(), outputGrad, this->weightsGrads,
              this->grad.get(), this->temp);
}

template <typename T>
void linearFwdOp(const Container<T> *inputs, const Container<T> *weights,
                 Container<T> *outputs) {
  operatorMatMul(inputs, weights, outputs);
}

template <typename T>
void linearBwdOp(
    const Container<T> *inputs, const Container<T> *weights,
    const Container<T> *outputGrad, Container<T> *weightsGrad,
    Container<T> *inputsGrad,
    std::unordered_map<std::string, std::unique_ptr<Container<T>>> &temp) {}

template <typename T>
void linearFwdBiasOp(const Container<T> *inputs, const Container<T> *biases,
                     const Container<T> *outputs) {
  // TODO
}

template <typename T>
void linearBwdBiasOp(const Container<T> *outputGrad, Container<T> *biasesGrad) {
  // TODO
}
