

#include "../Data/container.cuh"
template <typename T> class Optimizer {
public:
  virtual void step() = 0;
  virtual void
  reg(std::vector<std::pair<Container<T> *, Container<T> *>> params) = 0;

private:
  std::vector<Container<T> *> paramsList;
  std::vector<Container<T> *> gradsList;
};
