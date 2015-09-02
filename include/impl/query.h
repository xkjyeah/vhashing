#pragma once

namespace vhashing{
namespace detail {

struct AlwaysTrue {
  template <typename Key, typename T>
  inline __device__ __host__
  bool operator()(const Key &, const T&) const {
    return true;
  }
};

template<class HashTable, class Fil>
struct Filter_ {
  HashTable bm;
  Fil f;

  Filter_(const HashTable &bm, Fil f) : bm(bm), f(f) {}

#ifdef __CUDA_ARCH__
  inline __device__
#else
  inline __host__
#endif
  bool operator()(const typename HashTable::HashEntry &he) {
    return (!bm.isequal(he.key, bm.EmptyKey())
          && f(he.key, bm.alloc[he.block_index]));
  }

};

template <class HashTable, class Func>
struct Apply_ {
  HashTable ht;
  Func f;
  typedef typename HashTable::KeyType Key;
  typedef typename HashTable::HashEntry HashEntry;


#ifdef __CUDA_ARCH__
  __device__
#else
  __host__
#endif
  Apply_(HashTable ht, Func f)
  : ht(ht), f(f) {}

#ifdef __CUDA_ARCH__
  inline __device__
#else
  inline __host__
#endif
  void operator() (HashEntry &he) {
    if ( !ht.isequal(he.key, ht.EmptyKey()) ) {
      f(he.key, ht[he]);
    }
  }

};

/**
 * Use by AllocKeys for bulk allocations.
 *
 * */
template <class HashTable>
struct RequiresAllocation {
  HashTable bm;
  typedef typename HashTable::KeyType Key;
#ifdef __CUDA_ARCH__
  __device__
#else
  __host__
#endif
  bool operator() (const Key &k) {
    return !bm.isequal(k, bm.EmptyKey()) && bm.find(k) == bm.end();
  }
};

/**
 * For each key, try to allocate it in hashtable.
 *
 * Whether it fails/succeeds due to contention, record the result
 * in success.
 *
 * */
template <class HashTable>
__global__
void TryAllocateKernel(
    HashTable self,
    typename HashTable::KeyType *keys,
    int *success,
    int blockBase,
    int numJobs
    ) {
  typedef typename HashTable::ValueType T;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (x >= numJobs)
    return;
  
  T* result = self.real_insert(keys[x], T(), blockBase + x);

  if (result == 0) { // bucket conflict -- one wasted allocation
    success[x] = self.alloc.offsets[blockBase + x];
  }
  else {
    success[x] = -1;
  }
}

/**
 * If allocation failed due to contention for some keys,
 * then for each of these keys, free the allocated block,
 * and record which keys failed.
 * */
template <class BlockMap>
__global__
void ReturnAllocations(
    BlockMap self,
    int *success,
    int *unsuccessful,
    int blockBase,
    int numJobs) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (x >= numJobs)
    return;

  if (success[x] != -1) {
    *unsuccessful = 1;
    self.alloc.free(success[x]);
  }
}

}
}
