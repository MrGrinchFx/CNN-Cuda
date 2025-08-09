
#include "../utils.cuh"
#include "linear.cuh"
#include "operators.cuh"
#include <__clang_cuda_builtin_vars.h>
#include <thrust/fill.h>
template <typename T> Linear<T>::Linear(int inputChannels, int outputChannels) {

  this->inputChannels = inputChannels;
  this->outputChannels = outputChannels;
  // weights initialization
  this->weights.reset(new Container<T>({inputChannels, outputChannels}));
  this->weightGrads.reset(new Container<T>({inputChannels, outputChannels}));
  this->weights->xavierInit(inputChannels,
                            outputChannels); // random initialization
  // bias initializationlinear.cu
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

  INIT_CONTAINER(this->output, outputShape, T);

  linearFwdOp(input, this->weights.get(), this->output.get());
  linearFwdBiasOp(input, this->biases.get(), this->output.get());
}

template <typename T> void Linear<T>::backward() {
  const Container<T> *input = this->prev->getOutput();
  const Container<T> *outputGrad = this->next->getGrad();

  INIT_CONTAINER(this->grad, input->getShape(), T);

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
    std::unordered_map<std::string, std::unique_ptr<Container<T>>> &transpose) {
  // Weight Transpose
  std::vector<int> weightTransposeShape = {weights->getShape()[1],
                                           weights->getShape()[0]};
  INIT_TEMP(transpose, "weightsTran", weightTransposeShape, T);
  transposeOp(weights, transpose["weightsTran"].get());
  // Input Transpose

  std::vector<int> inputTransposeShape = {inputs->getShape()[1],
                                          inputs->getShape()[0]};
  INIT_TEMP(transpose, "inputsTran", inputTransposeShape, T);
  transposeOp(inputs, transpose["inputsTran"].get());

  // Output = X * W

  // dE/dX = dE/dY * W^T
  operatorMatMul(outputGrad, transpose["weightsTran"].get(), inputsGrad);

  // dE/dW = X^T * dE/dY
  operatorMatMul(transpose["inputsTran"].get(), outputGrad, weightsGrad);
}

template <typename T>
void __global__ operatorBias(const Container<T> *inputs,
                             const Container<T> *biases, Container<T> *outputs,
                             int size, int width) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    int col = index % width;
    outputs[index] = inputs[index] + biases[index];
  }
}

template <typename T>
void linearFwdBiasOp(const Container<T> *inputs, const Container<T> *biases,
                     const Container<T> *outputs) {
  // Add bias to the inputs
  const T *inputsPtr = RAW_PTR(inputs->getData());
  const T *biasesPtr = RAW_PTR(biases->getData());
  const T *outputsPtr = RAW_PTR(outputs->getData());

  int size = inputs->getData().size();
  int gridSize = ceil((float)(size) / BLOCK_SIZE);
  int width = biases->getData().size();
  // inputs + biases = outputs
  operatorBias<<<gridSize, BLOCK_SIZE>>>(inputsPtr, biasesPtr, outputsPtr, size,
                                         width);

  CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void linearBwdBiasOp(const Container<T> *outputGrad, Container<T> *biasesGrad) {
  // dE/dB = (dY/dB) 1 * dE/dY
  //(i.e dE/dB = dE/dY)
  operatorSum(outputGrad, 0, biasesGrad);
}
