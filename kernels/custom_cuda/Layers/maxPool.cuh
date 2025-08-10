#include "layers.cuh"
template <typename T> class MaxPool : public Layer<T> {
public:
  MaxPool(int kernelSize, int stride);

  void forward();
  void backward();

private:
};
