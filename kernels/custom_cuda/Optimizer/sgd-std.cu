#include "../utils.cuh"
#include "sgd-std.cuh"

template <typename T> void updateFunction() {}

template <typename T>
void sgdUpdate(Container<T> *weights, Container<T> *grads, float learningRate,
               float l2) {
  CHECK_EQ(weights->getData().size(), grads->getData().size(),
           "Weights and Gradient list shape doesn't match!");

  // update weights
  updateFunction<T>();
}

template <typename T> void SGD<T>::step() {
  for (int i = 0; i < this->paramsList.size(); i++) {
    sgdUpdate(this->paramsList[i], this->gradsList[i], this->learningRate,
              this->l2, this->beta);
  }
}

template <typename T>
void SGD<T>::reg(
    std::vector<std::pair<Container<T> *, Container<T> *>> params) {
  for (auto it = params.begin(); it != params.end(); it++) {
    this->paramsList.push_back(it->first);
    this->gradsList.push_back(it->second);
  }
}
