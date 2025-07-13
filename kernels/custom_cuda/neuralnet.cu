#include <cuda_device_runtime_api.h>
#include <iostream>
#include <vector>
template <typename T> class MLP {
private:
  std::vector<T> layers = {};

public:
  MLP() { std::cout << "TODO Constructor" << std::endl; }

  ~MLP() { std::cout << "TODO Destructor" << std::endl; }
  void train() {}
  void validate() {}
  void forward() {}
  void backward() {}
};
