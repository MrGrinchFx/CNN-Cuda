
#include "optimizer.cuh"

template <typename T>
void sgdFunction(Container<T> *weights, const Container<T> *grads,
                 float learning_rate = 1e-2, float l2 = 1e-3);

template <typename T> class SGD : public Optimizer<T> {
public:
  explicit SGD(float learningRate = 1e-2, float l2 = 1e-3)
      : learningRate(learningRate), l2(l2) {
    std::cout << "Learning Rate: " << learningRate << ", l2: " << l2
              << std::endl;
  }

  void reg(std::vector<std::pair<Container<T> *, Container<T> *>> params);
  void step();

private:
  float learningRate;
  float l2;
  float beta;
};
