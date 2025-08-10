#include "layers.cuh"
template <typename T> class Flatten : public Layer<T> {
public:
  Flatten<T>();
  void forward();
  void backward();

private:
};
