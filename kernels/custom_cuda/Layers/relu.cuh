#include "layers.cuh"
template <typename T> class Relu : public Layer<T> {
public:
  Relu<T>(int numInputs);
  void forward();
  void backward();

private:
};
