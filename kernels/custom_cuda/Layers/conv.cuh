#include "../Data/container.cuh"
#include "layers.cuh"
#include <unordered_map>
template <typename T> class Conv : public Layer<T> {
public:
  Conv(int inputChannels, int outputChannels, int kernelSize, int padding,
       int stride);
  void forward();
  void backward();

  std::vector<std::pair<Container<T> *, Container<T> *>> parameters;

private:
  std::unique_ptr<Container<T>> filters;
  std::unique_ptr<Container<T>> filterGrad;
  std::unique_ptr<Container<T>> bias;
  std::unique_ptr<Container<T>> biasGrad;
  std::unique_ptr<Container<T>> cols;

  std::unordered_map<std::string, std::unique_ptr<Container<T>>> temp;
  int inputChannels;
  int outputChannels;
  int kernelHeight;
  int kernelWidth;
  int paddingHeight;
  int paddingWidth;
  int strideHeight;
  int strideWidth;
  bool isBias;
};
