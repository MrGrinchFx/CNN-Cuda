#include "layers.hpp"
#include <unordered_map>
template <typename T> class Conv : Layer<T> {
public:
  explicit Conv<T>(int _inputChannels, int _outputChannels, int _kernelSize,
                   int _padding, int _stride)
      : inputChannels(_inputChannels), outputChannels(_outputChannels),
        kernelHeight(_kernelSize), kernelWidth(_kernelSize),
        paddingHeight(_padding), paddingWidth(_padding), strideHeight(_stride),
        strideWidth(_stride) {}
  void forward();
  void backward();

  std::vector<std::pair<Storage *, Storage *>> parameters;

private:
  std::unique_ptr<Storage> filters;
  std::unique_ptr<Storage> filterGrad;
  std::unique_ptr<Storage> bias;
  std::unique_ptr<Storage> biasGrad;
  std::unique_ptr<Storage> cols;

  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
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
