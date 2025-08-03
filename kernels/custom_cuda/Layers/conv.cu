#include "conv.cuh"

template <typename T>
Conv<T>::Conv(int _inputChannels, int _outputChannels, int _kernelSize,
              int _padding, int _stride)
    : inputChannels(_inputChannels), outputChannels(_outputChannels),
      kernelHeight(_kernelSize), kernelWidth(_kernelSize),
      paddingHeight(_padding), paddingWidth(_padding), strideHeight(_stride),
      strideWidth(_stride) {}
template <typename T> void Conv<T>::forward() {}
template <typename T> void Conv<T>::backward() {}
