#include "../utils.cuh"
#include "sgd-std.cuh"

struct sgdUpdateFunctor {
  const float lr;
  const float l2;

  sgdUpdateFunctor(float lr, float l2) : lr(lr), l2(l2) {}

  template <typename Tuple> __host__ __device__ void operator()(Tuple t) {
    // get the weight and gradient from the tuple
    float &weight = thrust::get<0>(t);
    float grad = thrust::get<1>(t);

    // apply L2 regularization
    grad += l2 * weight;

    // apply the SGD update rule
    weight -= lr * grad;
  }
};

template <typename T>
void sgdFunction(Container<T> *weights, const Container<T> *grads,
                 float learningRate, float l2) {
  CHECK_EQ(weights->getData().size(), grads->getData().size(),
           "Weights and Gradient list shape doesn't match!");

  // update weights
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                       weights->getData().begin(), grads->getData().begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(
                       weights->getData().end(), grads->getData().end())),
                   sgdUpdateFunctor(learningRate, l2));
}

template <typename T> void SGD<T>::step() {
  for (int i = 0; i < this->paramsList.size(); i++) {
    sgdFunction(this->paramsList[i], this->gradsList[i], this->learningRate,
                this->l2);
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
