
#include "../utils.hpp"
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
Dataloader<T>::Dataloader(std::string dataPath, bool shuffle)
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

template <typename T> void Dataloader<T>::forward(int batchSize, bool isTrain) {
  if (isTrain) {
    int startIdx = trainIndex;
    int endIdx = trainIndex + std::min(this->trainIndex + batchSize,
                                       (int)this->trainData.size());

    this->trainIndex = endIdx;
    int size = endIdx - startIdx;

    // init device memory
    std::vector<int> shape{size, 1, this->height, this->width};
    std::vector<int> labelShape{size, 10};

    // LEARN!!!!
    INIT_CONTAINER(this->output, shape);
    INIT_CONTAINER(this->outputLabel, labelShape);

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
                             this->trainData.end());
      this->outputLabel
          ->getData()[(i - startIdx) * labelStride + this->trainLabel[i]] = 1;
    }

    this->output->getData() = trainDataBuffer;
  } else {
    int startIdx = testIndex;
    int endIdx = std::min(testIndex + batchSize, (int)this->testData.size());

    this->testIndex = endIdx;
    int size = endIdx - startIdx;

    std::vector<int> shape{size, 1, this->height, this->width};
    std::vector<int> labelShape{size, 10};

    INIT_CONTAINER(this->output, shape);
    INIT_CONTAINER(this->outputLabel, labelShape);

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
