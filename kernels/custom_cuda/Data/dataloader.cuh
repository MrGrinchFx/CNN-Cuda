#pragma once

#include "../Layers/layers.cuh"
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

template <typename T> class Dataloader : public Layer<T> {
public:
  Dataloader(std::string dataPath, bool shuffle = true, int batchSize = 64);
  void forward();
  bool nextBatchIsHere(bool isTrain);
  void reset();
  int getHeight() { return this->height; }
  int getWidth() { return this->width; }
  Container<unsigned char> *getLabel() { return this->outputLabel.get(); }
  void printImg();
  unsigned int convertBigEndian(unsigned int i);

private:
  void readImgs(std::string fileName, std::vector<std::vector<T>> &images);
  void readLabels(std::string fileName, std::vector<unsigned char> &labels);
  std::vector<std::vector<T>> trainData;
  std::vector<std::vector<T>> testData;
  std::vector<unsigned char> trainLabel;
  std::vector<unsigned char> testLabel;
  int testIndex;
  int trainIndex;
  int batchSize;
  int height;
  int width;
  bool shuffle;
  bool isTrain;
  std::unique_ptr<Container<unsigned char>> outputLabel;
  std::unique_ptr<Container<T>> output;
};
