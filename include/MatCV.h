#pragma once

#include <opencv2/opencv.hpp>
#include <memory>
#include "MatWrapper.h"

namespace cv { class Mat; }

namespace vhashing {

/** wraps around OpenCV matrices, whilst maintaining a reference to
 * the data. WARNING: Do not make a pointer of this class! The deleter
 * of MatBase is not virtual.
 * **/
template <typename T>
struct MatCV : public MatBase<T> {
  std::shared_ptr<T> ref2;
  cv::Mat ref;

  MatCV(const cv::Mat &img)
    : MatBase<T>(img.rows, img.cols, (T*)img.data, img.step1() * img.elemSize1()),
      ref(img) {
    assert(sizeof(T) == img.elemSize());
  }

  MatCV(size_t rows, size_t cols)
  : 
    MatBase<T>(rows, cols, 0, cols * sizeof(T)),
    ref2(new T[rows * cols], [](T* t) { delete[] t; })
  {
    this->data_ = ref2.get();
  }
};


}

