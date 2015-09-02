#pragma once
#include "MatWrapper.h"
#include <memory>
#include <cuda_runtime.h>
#include "safecall.h"

namespace vhashing {
/** 
 * a wrapper matrix to cast MatBase types to device types
 *
 * The device memory allocation is not freed until the MatDevice is destroyed.
 *
 * You can pass this device as a MatBase<PixelType> to __global__ functions, 
 * so that the copy does not free the memory.
 *
 * */
template <typename PixelType>
struct MatDevice : public MatBase<PixelType> {
  typedef MatBase<PixelType> ParentType;

  struct cuda_deleter_ {
    void operator() (const PixelType *p) {
      if (p) cudaSafeCall(cudaFree((void*)p));
    }
  };

  std::unique_ptr<PixelType, cuda_deleter_> dev_data;

  __host__
  MatDevice() {}
  __host__
  MatDevice(const ParentType &b, bool alignedMemory = false)
  : ParentType(b) {
    // allocate data
    if (alignedMemory) {
      cudaSafeCall(cudaMallocPitch((void**)&this->data_,
                        &this->stride_,
                        this->cols * sizeof(PixelType),
                        this->rows));
    }
    else {
      cudaSafeCall(cudaMalloc((void**)&this->data_,
                          this->cols * sizeof(PixelType) *
                          this->rows));
      this->stride_ = this->cols * sizeof(PixelType);
    }
    cudaSafeCall(cudaMemcpy2D(this->data_,
                              this->stride_,
                              b.data_,
                              b.stride_,
                              this->cols * sizeof(PixelType),
                              this->rows,
                              cudaMemcpyHostToDevice));

    dev_data.reset((PixelType*)this->data_);
  }
};

/** Equivalent class as MatDevice, but from device
 * to host **/
template <typename PixelType>
struct MatHost : public MatBase<PixelType> {
  typedef MatBase<PixelType> ParentType;

  __host__
  MatHost(const ParentType &b)
  : ParentType(b) {
    // allocate data -- no padding
    this->data_ = (uint8_t*)malloc(this->cols * sizeof(PixelType) * this->rows);
    this->stride_ = sizeof(PixelType) * this->cols;

    cudaSafeCall(cudaMemcpy2D(this->data_,
                              this->stride_,
                              b.data_,
                              b.stride_,
                              this->cols * sizeof(PixelType),
                              this->rows,
                              cudaMemcpyDeviceToHost));
  }

  __host__
  ~MatHost() {
    free(this->data_);
  }
};
}
