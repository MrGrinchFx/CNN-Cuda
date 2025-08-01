#include "linear.cuh"

template <typename T> Linear<T>::Linear(int inputChannels, int outputChannels) {
  this->inputChannels = inputChannels;
  this->outputChannels = outputChannels;
}
