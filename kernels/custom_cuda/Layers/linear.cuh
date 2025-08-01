#include "../Data/container.cuh"
#include "layers.cuh"
#include <memory>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>
template <typename T> class Linear : Layer<T> {
public:
  explicit Linear(int inputChannels, int outputChannels);
  void forward();
  void backward();

  std::vector<std::pair<Container<T> *, Container<T> *>> parameters;

private:
  std::unique_ptr<Container<T>> weights;
  std::unique_ptr<Container<T>> weightGrads;
  std::unique_ptr<Container<T>> bias;
  std::unique_ptr<Container<T>> biasGrads;

  std::unordered_map<std::string, std::unique_ptr<Container<T>>> temp;

  int inputChannels;
  int outputChannels;
};
