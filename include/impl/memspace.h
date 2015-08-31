#pragma once

#include "../safecall.h"
#include <thrust/host_vector.h>

namespace vhashing {

struct host_memspace {
};
struct std_memspace {
};
struct device_memspace {
};

template <typename T, typename M>
struct ptr_type {};

template <typename T>
struct ptr_type<T, host_memspace> {
  typedef T* type;
};
template <typename T>
struct ptr_type<T, std_memspace> {
  typedef T* type;
};
template <typename T>
struct ptr_type<T, device_memspace> {
  typedef thrust::device_ptr<T> type;
};

template <typename T, typename M>
struct vector_type {};

template <typename T>
struct vector_type<T, host_memspace> {
  typedef thrust::host_vector<T> type;
};
template <typename T>
struct vector_type<T, std_memspace> {
  typedef std::vector<T> type;
};
template <typename T>
struct vector_type<T, device_memspace> {
  typedef thrust::device_vector<T> type;
};


  namespace detail {

template <typename T>
T* memspace_alloc(size_t num_elems, host_memspace) {
  T* t = malloc(num_elems * sizeof(T));
  assert(t);
  return t;
}

template <typename T>
T* memspace_alloc(size_t num_elems, device_memspace) {
  T* t = 0;
  cudaSafeCall(cudaMalloc(&t, num_elems * sizeof(T)));
  return t;
}

template <typename T>
void memspace_fill(T* start, T* end, const T &t, host_memspace) {
  thrust::uninitialized_fill(start, end, t);
}

template <typename T>
void memspace_fill(T* start, T* end, const T &t, device_memspace) {
  thrust::uninitialized_fill(
      thrust::device_pointer_cast(start),
      thrust::device_pointer_cast(end),
      t);
}


template <typename T>
struct memspace_deleter {};

template <>
struct memspace_deleter<host_memspace> {
  template <typename T>
  void operator() (const T *t) {
    free(t);
  }
};

template <>
struct memspace_deleter<device_memspace> {
  template <typename T>
  void operator() (const T *t) {
    cudaSafeCall(cudaFree(t));
  }
};

}  // namespace detail
}  // namespace vhashing
