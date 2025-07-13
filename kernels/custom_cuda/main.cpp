#include "neuralnet.hpp"
#include <iostream>

using std::cout, std::endl;

int main() {
  // Starting point of the include
  // Populate the Weights and biases with random numbers (Can be of any type)
  // Call the constructor for an architecture object (it will accept a vector of
  // layers and activations) Call a MLP.train() that will conduct training (Will
  // define training logic in a separate file)
  Architecture<int> myModel;
  int numClasses = 10; // number of different digits in MNIST

  // example usage | check ~/python/python_reference.ipynb for the pytorch
  // equivalent of this model!
  myModel.addConv(1, 32, 5, 2, 1);
  myModel.addMaxPool(2, 2);
  myModel.addConv(32, 64, 5, 1, 2);
  myModel.addLinear(14 * 14 * 64, 1024);
  myModel.addRelu();
  myModel.addLinear(1024, numClasses);
  myModel.train();

  return 0;
}
