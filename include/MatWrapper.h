#pragma once 

#include <cuda_runtime.h>

namespace vhashing {

/**
 * A wrapper matrix without ref counting */
template <typename PixelType>
struct MatBase {
  uint8_t *data_;
	size_t rows, cols, stride_;
	size_t size_;

  /* stride in bytes */
	__host__
	MatBase(size_t prows, size_t pcols, PixelType *data = nullptr, size_t stride = 0)
	: data_((uint8_t*)data),
    rows(prows),
    cols(pcols),
    stride_(std::max(cols * sizeof(PixelType), stride))
	{
    size_ = rows * stride_;
	}

	__host__
	MatBase()
	:data_(0), rows(0), cols(0), stride_(0)
	{}

	__device__ __host__
	inline
	MatBase(const MatBase &) = default;

	__device__ __host__
	inline
	PixelType &at(int2 xy) const {
		return at(xy.y, xy.x);
	}
	/* compatibility with cv::MatBase::at */
	__device__ __host__
	inline
	PixelType &at(int y, int x) const {
		return *reinterpret_cast<PixelType*>(data_ + y*stride_ + x*sizeof(PixelType));
	}
	__device__ __host__
	inline
	PixelType &operator()(int y, int x) const {
		return *reinterpret_cast<PixelType*>(data_ + y*stride_ + x*sizeof(PixelType));
	}

	__device__ __host__
	inline
	~MatBase() {
	}
};

}

