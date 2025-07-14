#include "neuralnet.hpp"
#include "utils.hpp"
#include <iostream>

using std::cout, std::endl;

// data loader
std::vector<uint8_t> load_mnist_images(const std::string &filename,
                                       int num_images, int image_size) {
  std::ifstream file(filename, std::ios::binary);
  file.ignore(16); // skip header
  std::vector<uint8_t> images(num_images * image_size);
  file.read(reinterpret_cast<char *>(images.data()), images.size());
  return images;
}

std::vector<uint8_t> load_mnist_labels(const std::string &filename,
                                       int num_labels) {
  std::ifstream file(filename, std::ios::binary);
  file.ignore(8); // skip header
  std::vector<uint8_t> labels(num_labels);
  file.read(reinterpret_cast<char *>(labels.data()), labels.size());
  return labels;
}

int main() {
  // Starting point of the include
  // Populate the Weights and biases with random numbers (Can be of any type)
  // Call the constructor for an architecture object (it will accept a vector of
  // layers and activations) Call a MLP.train() that will conduct training (Will
  // define training logic in a separate file)
  Architecture<float> myModel;
  int numClasses = 10; // number of different digits in MNIST

  load_mnist_images("../../data/MNIST/raw/train-images-idx3-ubyte", 60000, 784);
  load_mnist_labels("../../data/MNIST/raw/train-labels-idx1-ubyte", 60000);
  load_mnist_images("../../data/MNIST/raw/t10k-images-idx3-ubyte", 10000, 784);
  load_mnist_labels("../../data/MNIST/raw/t10k-labels-idx1-ubyte", 10000);
  // example usage | check ~/python/python_reference.ipynb for the pytorch
  // equivalent of this model!
  myModel.addConv(1, 32, 5, 2, 1);
  myModel.addMaxPool(2, 2);
  myModel.addConv(32, 64, 5, 1, 2);
  myModel.addLinear(14 * 14 * 64, 1024);
  myModel.addRelu();
  myModel.addLinear(1024, numClasses);
  myModel.train();
  myModel.printResults();

  return 0;
}
