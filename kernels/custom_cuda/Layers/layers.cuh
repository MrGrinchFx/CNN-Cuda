#pragma once
#include "../Data/container.cu"
#include <memory>
#include <vector>
template <typename T> class Layer {
  // Disallow any copy or move constructors that would break our model.
  Layer() {}
  Layer(const Layer &other) = delete;
  Layer(Layer &&other) = delete;
  Layer &operator=(const Layer &other) = delete;
  Layer &operator=(Layer &&other) = delete;

public:
  Layer &connect(Layer &nextLayer) {
    this->next = &nextLayer;
    nextLayer.prev = this;

    return nextLayer;
  }

  virtual void forward();
  virtual void backward();
  virtual std::vector<std::pair<Container<T> *, Container<T> *>>
  getParameters();
  virtual Container<T> *getGrad() { return this->grad.get(); };
  virtual Container<T> *getOutput() { return this->output.get(); };

private:
  Layer *next = nullptr;
  Layer *prev = nullptr;

  // a neighbor layer would call next->getGrad() to retrieve grad information
  // for the backward prop
  std::unique_ptr<Container<T>> grad;
  // a neigbor layer would call prev->getOutput() to retrieve output information
  // for the forward prop
  std::unique_ptr<Container<T>> output;
};
