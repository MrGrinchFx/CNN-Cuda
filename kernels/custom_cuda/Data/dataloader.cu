
#include "../utils.cuh"
#include "bits/stdc++.h"
#include "dataloader.cuh"
#include "thrust/host_vector.h"
#include "thrust/mr/allocator.h"
#include "thrust/system/cuda/memory_resource.h"
#include <chrono>
#include <fstream>
#include <random>
#include <vector>

template <typename T>
Dataloader<T>::Dataloader(std::string dataPath, bool shuffle, int batchSize)
    : shuffle(shuffle), trainIndex(0), testIndex(0) {
  this->readImgs(dataPath + "/train-images-idx3-ubyte", this->trainData);
  this->readLabels(dataPath + "/train-labels-idx1-ubyte", this->trainLabel);

  this->readImgs(dataPath + "/test-images-idx3-ubyte", this->testData);
  this->readLabels(dataPath + "/test-labels-idx1-ubyte", this->testLabel);
}

template <typename T> void Dataloader<T>::reset() {
  this->trainIndex = 0;
  this->testIndex = 0;

  if (shuffle) {
    unsigned int seed =
        std::chrono::system_clock::now().time_since_epoch().count() % 1234;

    std::shuffle(this->trainData.begin(), this->trainData.end(),
                 std::default_random_engine(seed));
    std::shuffle(this->trainLabel.begin(), this->trainLabel.end(),
                 std::default_random_engine(seed));
  }
}

template <typename T> void Dataloader<T>::forward() {
  if (isTrain) {
    int startIdx = trainIndex;
    int endIdx = std::min(startIdx + batchSize, (int)this->trainData.size());

    this->trainIndex = endIdx;
    int size = endIdx - startIdx;

    // init device memory
    std::vector<int> shape{size, 1, this->height, this->width};
    std::vector<int> labelShape{size, 10};

    // LEARN!!!!
    INIT_CONTAINER(this->output, shape, T);
    INIT_CONTAINER(this->outputLabel, labelShape, T);

    thrust::fill(this->outputLabel->getData().begin(),
                 this->outputLabel->getData().end(), 0);

    int imgStride =
        this->height * this->width; // height * width pixels for each img
    int labelStride = 10;           // 10 labels per img for each class

    // review how the following works :: LEARN!!!!
    thrust::host_vector<
        T, thrust::mr::allocator<
               T, thrust::system::cuda::universal_host_pinned_memory_resource>>
        trainDataBuffer;
    trainDataBuffer.reserve(size * imgStride);

    // load each image and label in the batch to the allocated buffer.
    for (int i = startIdx; i < endIdx; i++) {
      trainDataBuffer.insert(trainDataBuffer.end(), this->trainData[i].begin(),
                             this->trainData[i].end());
      this->outputLabel
          ->getData()[(i - startIdx) * labelStride + this->trainLabel[i]] = 1;
    }

    this->output->getData() = trainDataBuffer;
  } else {
    int startIdx = testIndex;
    int endIdx = std::min(startIdx + batchSize, (int)this->testData.size());

    this->testIndex = endIdx;
    int size = endIdx - startIdx;

    std::vector<int> shape{size, 1, this->height, this->width};
    std::vector<int> labelShape{size, 10};

    INIT_CONTAINER(this->output, shape, T);
    INIT_CONTAINER(this->outputLabel, labelShape, T);

    thrust::fill(this->outputLabel->getData().begin(),
                 this->outputLabel->getData().end(), 0);

    int imgStride = this->height * this->width;
    int labelStride = 10;

    thrust::host_vector<
        T, thrust::mr::allocator<
               T, thrust::system::cuda::universal_host_pinned_memory_resource>>
        testDataBuffer;
    testDataBuffer.reserve(size * imgStride);

    for (int i = startIdx; i < endIdx; i++) {
      testDataBuffer.insert(testDataBuffer.end(), this->testData[i].begin(),
                            this->testData[i].end());
      this->outputLabel
          ->getData()[(i - startIdx) * labelStride + this->testLabel[i]] = 1;
    }

    this->output->getData() = testDataBuffer;
  }
}
template <typename T> bool Dataloader<T>::nextBatchIsHere(bool isTrain) {
  if (isTrain) {
    return this->trainDataIdx < this->trainData.size();
  } else {
    return this->testDataIdx < this->testData.size();
  }
}

template <typename T>
unsigned int Dataloader<T>::convertBigEndian(unsigned int i) {
  unsigned char b1, b2, b3, b4;
  b1 = i % 0xFF;
  b2 = (i >> 8) % 0xFF;
  b3 = (i >> 16) % 0xFF;
  b4 = (i >> 24) % 0xFF;

  return ((unsigned int)b1 << 24) + ((unsigned int)b2 << 16) +
         ((unsigned int)b3 << 8) + (b4);
}
template <typename T>
void Dataloader<T>::readImgs(std::string fileName,
                             std::vector<std::vector<T>> &imgs) {
  std::ifstream file(fileName, std::ios::binary);
  if (file.is_open()) {
    unsigned int magic;
    unsigned int numImgs;
    unsigned int numRows;
    unsigned int numCols;

    file.read((char *)&magic, sizeof(magic));
    file.read((char *)&numImgs, sizeof(numImgs));
    file.read((char *)&numRows, sizeof(numRows));
    file.read((char *)&numCols, sizeof(numCols));
    magic = this->convertBigEndian(magic);
    numImgs = this->convertBigEndian(numImgs);
    numRows = this->convertBigEndian(numRows);
    numCols = this->convertBigEndian(numCols);

    // print out into console for confirmation
    std::cout << fileName << "\n";
    std::cout << "Number of Images Loaded: " << numImgs << "\n";
    std::cout << "Number of (Rows, Cols): (" << numRows << ", " << numCols
              << ")\n";

    this->height = numRows;
    this->width = numCols;

    std::vector<unsigned char> image(numRows * numCols);
    std::vector<T> normalizedImages(numRows * numCols * sizeof(T));

    for (int i = 0; i < numImgs; i++) {
      file.read((char *)&image[0], sizeof(unsigned char) * numRows * numCols);
      for (int j = 0; j < numCols; j++) {
        normalizedImages[i] = (T)image[i] / 255 - 0.5f;
      }
      imgs.push_back(normalizedImages);
    }
  }
}

template <typename T>
void Dataloader<T>::readLabels(std::string fileName,
                               std::vector<unsigned char> &labels) {
  std::ifstream file(fileName, std::ios::binary);

  if (file.is_open()) {
    unsigned int magic = 0;
    unsigned int numImgs = 0;
    file.read((char *)&magic, sizeof(magic));
    file.read((char *)&numImgs, sizeof(numImgs));

    magic = this->convertBigEndian(magic);
    numImgs = this->convertBigEndian(numImgs);

    // print output for confirmation
    std::cout << "Loading Labels: " << fileName << "\n";
    std::cout << "Number of labels: " << numImgs << "\n";

    for (int i = 0; i < numImgs; i++) {
      unsigned char label = 0;
      file.read((char *)&label, sizeof(label));
      labels.push_back(label);
    }
  }
}

template <typename T> void Dataloader<T>::printImg() {
  // TODO
}
