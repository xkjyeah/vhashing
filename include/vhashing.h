#pragma once

#include "impl/lockset.h"
#include "impl/blockalloc.h"
#include "impl/memspace.h"

#include <memory>

namespace vhashing {

/**
 * Hash table with key type Key
 *
 * */
template <typename Key>
struct HashEntryBase {
  Key			      key;
  int32_t		    offset;
  int32_t    	block_index;
};


/**
 *
 * The underlying base class that can be accessed directly by __device__ code
 *
 * */
template <class Key,
          class Value,
          class Hash,
          class Equal>
struct HashTableBase {
  struct iterator;

  int     num_buckets;
  int     entries_per_bucket;
  uint32_t num_entries;

  Key emptyKey;
  Hash hasher;
  Equal isequal;

  typedef HashEntryBase<Key> HashEntry;
  typedef detail::BlockAllocBase<Value> BlockAlloc;

  HashEntry   *hash_table;
  int         *bucket_locks;

  BlockAlloc alloc;

  HashTableBase(int num_buckets,
                int entries_per_bucket,
                int num_blocks,
                Key emptyKey,
                Hash hasher = Hash(),
                Equal equals = Equal())
  : num_buckets(num_buckets),
    entries_per_bucket(entries_per_bucket),
    num_entries(num_buckets * entries_per_bucket),

    emptyKey(emptyKey),
    hasher(hasher),
    isequal(equals)
  {

  }


	__device__ __host__
	inline Key EmptyKey() const {
		return emptyKey;
	}

	__device__ __host__
	inline bool IsEmpty(int32_t off) const {
		return hash_table[off].key == EmptyKey();
	}

	__device__ __host__
	void error(int errcode, const char *errstring) {
#ifndef __CUDA_ARCH__
		throw "Error here!";
#else
		printf("Error at %s\n", errstring);
    *((int*)0) = 0xdeadbeef;
    assert(false);
#endif
	}

  /**
   * Access the value associated with he.key
   *
   * Version that takes a HashEntry. Because it
   * already has the pointer to the allocated block, it doesn't
   * require a search
   * */
	__device__ __host__
	Value &operator[](const HashEntry &he) const {
		return alloc.blocks[he.block_index];
	}
	/** Access the value associated with key k.
   *
   * Read-only version -- does not do any locking */
	__device__ __host__
	Value &operator[](const Key &k) const {
		auto it = this->tryfind(k, true);

    if (it == end()) {
      assert(false);
      return *((Value*)0);
    }
    return alloc[it->block_index];
  }
	/* for internal use -- where thread matters
	 *
	 * returns end() if not found.
	 * returns fail() if bucket has changed since we last inspected it
	 *
	 * */
	__device__ __host__
	iterator tryfind(const Key &k, bool readonly = false) const {
		uint64_t hash = hasher(k);
		int32_t bucket = hash % num_buckets;
		detail::LockSet<2> lockset;

		if (!readonly && !lockset.TryLock(bucket_locks[bucket]))
		{
			lockset.YieldAll();
			return fail();
		}
		
		// search bucket for the entry
		for (int i=0; i<entries_per_bucket; i++) {
			int32_t offset = bucket*entries_per_bucket + i;
			HashEntry &entr = hash_table[offset];

			if (isequal(entr.key, k)) {
				return iterator(*this, offset);
			}
		}

		// search linked list
		int32_t offset = bucket*entries_per_bucket + entries_per_bucket - 1;
		while (hash_table[offset].offset) {
			HashEntry &entr = hash_table[offset];
			int nextBucket = ((offset + entr.offset) % num_entries) / entries_per_bucket;

			if (!readonly && !lockset.TryLock(bucket_locks[nextBucket]))
			{
				lockset.YieldAll();
				return fail();
			}

			HashEntry &next_entr = hash_table[(offset + entr.offset) % num_entries];
			if (isequal(next_entr.key, k)) {
				return iterator(*this, (offset + entr.offset) % num_entries);
			}

			lockset.Yield(bucket_locks[offset / entries_per_bucket]);
			offset = (offset + entr.offset) % num_entries;
		}

		// not found and no more links to visit
		return end();
	}

	/**
   * Access the value associated with key k.
   *
   * It is not recommended to use this for performance reasons.
   * Instead, try breaking up your algorithm into two kernels --
   * one to allocate, and one to assign value
   *
   * Write-version -- must handle concurrency.
	 * FIXME: two locks -- read lock / write lock for speed */
	__device__ __host__
	Value &operator[](const Key &k) {
		/* maintain a lock on the bucket -- so
			only one insertion takes place at any one time */
		//uint64_t hash = hasher(k);
		//int32_t bucket = hash % num_buckets;
		Value* rv;
		bool done = false;

		// while (!done) because of CUDA Warp paradigm
		// FIXME: The probability of livelock?
		while (!done) {
			iterator it = tryfind(k);

			if (it == fail()) {
				// do nothing
			}
			else if (it != end()) {
				rv = &alloc[it->block_index];
				done = true;
			}
			else /* it == end */ {
				rv = real_insert(k, Value());
				if (rv)
					done = true;
			}
		}
		if (rv) {
			return *rv;
		}
		else {
      assert(false);
			return *((Value*)0);
		}
	}

	/**
	 * a la Niessner: "if a bucket is locked
	 * for writing all other allocations for the same bucket
	 * are staggered until the next frame is processed"
	 * 
	 * i.e. give up!
	 * */
	__device__ __host__
	Value *tryinsert(const Key &k, const Value &t = Value()) {
		/* maintain a lock on the bucket -- so
			only one insertion takes place at any one time */
		Value* rv;

		iterator it = tryfind(k);

		if (it == fail()) {
			rv = 0;
		}
		else if (it != end()) {
			rv = &alloc[it->block_index];
		}
		else /* it == end */ {
			rv = real_insert(k, t);
		}
		if (rv) {
			return &rv;
		}
		else {
			return 0;
		}
	}


  /**
   * Erase the key k
   * */
	__device__ __host__
	size_t erase(const Key &k) {
		detail::LockSet<2> lockset;
		bool done = false;
		int rv;

		while (!done) {
			if ( (rv = tryerase(k, lockset)) != (int)-1) {
				done = true;
			}
			lockset.YieldAll();
		}
		return rv;
	}

  /**
   * Attempt to erase key k
   *
   * Returns -1 if failed to acquire lock
   * 1 if successful
   * 0 if key does not exist
   *
   * */
	__device__ __host__
	int tryerase(const Key &k, detail::LockSet<2> &lockset) {
    /**
     * Must always have two locks -- the lock of the bucket being referenced,
     * and the lock on the referencing target
     *
     * So when the referenced bucket is erased, the referencing target can
     * update its offset without fear
     * */
		uint64_t hash = hasher(k);
		int32_t bucket = hash % num_buckets;

		if (!lockset.TryLock(bucket_locks[bucket]))
			return (int)-1;

		// search bucket for the entry
		for (int i=0; i<entries_per_bucket; i++) {
			int32_t offset = bucket*entries_per_bucket + i;
			HashEntry &entr = hash_table[offset];

			if (isequal(entr.key, k)) {
        alloc[entr.block_index].~Value();
				alloc.free(entr.block_index);

				if (i == entries_per_bucket - 1 && entr.offset) {
					HashEntry &next_entr = hash_table[(offset + entr.offset) % num_entries];
					int link_bucket = ((offset + entr.offset) % num_entries) / entries_per_bucket;
					if (!lockset.TryLock(bucket_locks[link_bucket])) {
						return (int)-1;
					}

					entr.key = next_entr.key;
					entr.block_index = next_entr.block_index;
					entr.offset = (next_entr.offset) ? (next_entr.offset + entr.offset)
																						: 0;
					
					next_entr.key = EmptyKey();
					next_entr.block_index = 0;
					next_entr.offset = 0;
				}
				else {
					entr.block_index = 0;
					entr.key = EmptyKey();
					entr.offset = 0;
				}
				return 1;
			}
		}

		// bucket overflow -- add to linked list
		int32_t offset = bucket*entries_per_bucket + entries_per_bucket - 1;
		while (hash_table[offset].offset) {
			HashEntry &entr = hash_table[offset];

			// lock to previous should already exist

			// lock to the next bucket
			int linkB = ((offset + entr.offset) % num_entries) / entries_per_bucket;
			if (!lockset.TryLock(bucket_locks[linkB])) {
				return (int)-1;
			}

			HashEntry &next_entr = hash_table[(offset + entr.offset) % num_entries];

			if (isequal(next_entr.key, k)) {
				alloc.free(next_entr.block_index);
				
				next_entr.block_index = 0;
				next_entr.key = EmptyKey();
				
				entr.offset = (next_entr.offset) ? 
								(entr.offset + next_entr.offset) : 0;
				
				next_entr.offset = 0;
				return 1;
			}

			// release the lock to previous
			lockset.Yield(bucket_locks[offset / entries_per_bucket]);

			offset = (offset + entr.offset) % num_entries;
		}
		return 0;
	}

	private:
	/**
	 * searches for an empty slot and inserts the item
	 * 
	 * Returns a NULL pointer if it failed (e.g. due to concurrency issues)
	 *
	 * Returns the item otherwise.
	 *
	 * */
	__device__ __host__
	Value *real_insert(const Key &k, const Value &val, int block_index = -1) {
		uint64_t hash = hasher(k);
		int32_t bucket = hash % num_buckets;

		// lock bucket
		detail::LockSet<2> lockset;
		if (!lockset.TryLock(bucket_locks[bucket]))
			return 0;

		// only one thread is here per bucket. :)
		// however multiple readers may exist that are frantically looking for a matching key

		// search bucket for the entry
		for (int i=0; i<entries_per_bucket; i++) {
			int32_t offset = bucket*entries_per_bucket + i;
			HashEntry &entr = hash_table[offset];

			if (isequal(entr.key, EmptyKey())) {
				entr.key = k;
				entr.block_index = (block_index == -1) ? alloc.allocate() : alloc.offsets[block_index];
				entr.offset = 0;
				new (&alloc[entr.block_index]) Value(val);

				return &alloc[entr.block_index];
			}
		}

		// bucket overflow -- add to linked list
		int32_t offset = bucket*entries_per_bucket + entries_per_bucket - 1;
		while (hash_table[offset].offset) {
			/* this set of lock acquisitions / yields protect against
			 * operations on other buckets */
			int nextBucket = ((offset + hash_table[offset].offset) % num_entries) / entries_per_bucket;
			if (!lockset.TryLock(bucket_locks[nextBucket]))
				return 0;
			lockset.Yield(bucket_locks[offset / entries_per_bucket]);

			offset = (offset + hash_table[offset].offset) % num_entries;
		}

		// found the list tail
		int32_t last_bucket = (int32_t) -1;
		for (short rel_offset = 1;
					rel_offset < 0x7FFF;
					rel_offset++) {
			int32_t real_offset = (offset + rel_offset) % num_entries;

			int32_t next_bucket = real_offset / entries_per_bucket;
			if (next_bucket != last_bucket) {
				if (last_bucket != (int32_t) -1)
					lockset.Yield(bucket_locks[last_bucket]);

				if (!lockset.TryLock(bucket_locks[next_bucket]))
					return 0;
				last_bucket = next_bucket;
			}

			if (isequal(hash_table[real_offset].key, EmptyKey())) {
				hash_table[real_offset] = {
					k,
					0,
					((block_index == -1)?alloc.allocate():alloc.offsets[block_index])
				};
				new (&alloc[hash_table[real_offset].block_index]) Value(val);
				hash_table[offset].offset = rel_offset;
				return &alloc[hash_table[real_offset].block_index];
			}
		}

    return 0;
	}

  public:

  /**
   * Iterator to access entries
   *
   * **/
	struct iterator {
		const HashTableBase &bm;
		int32_t offset;

		inline 	__device__ __host__
		iterator(const HashTableBase &bm, int32_t offset)
		: bm(bm), offset(offset)
		{}

		__device__ __host__
		inline HashEntry *operator->() const {
			return &bm.hash_table[offset];
		}
		__device__ __host__
		inline HashEntry &operator*() const {
			return bm.hash_table[offset];
		}
		__device__ __host__
		inline bool operator==(const iterator &b) const {
			return ( &bm == &b.bm && offset == b.offset );
		}
		__device__ __host__
		inline bool operator!=(const iterator &b) const {
			return !(*this == b);
		}
	};
	__device__ __host__
	inline iterator end() const {
		return iterator(*this, (int32_t) -1);
	}
	__device__ __host__
	inline iterator fail() const {
		return iterator(*this, (int32_t) -2);
	}

};

/**
 * Class for creating and passing structures around in host.
 *
 * */
template <class Key,
          class Value,
          class Hash,
          class Equal,
          class memspace = device_memspace>
class HashTable : public HashTableBase<Key,Value,Hash,Equal> {
  private:
  typedef HashTableBase<Key,Value,Hash,Equal> parent_type;

  typename vector_type<typename parent_type::HashEntry, memspace>::type hash_table_shared;
  typename vector_type<int, memspace>::type bucket_locks_shared;

  typename vector_type<Value, memspace>::type    data_shared;
  typename vector_type<int32_t, memspace>::type  offsets_shared;

  public:
  HashTable(int num_buckets,
                int entries_per_bucket,
                int num_blocks,
                Key emptyKey,
                Hash hasher = Hash(),
                Equal equals = Equal())
  : parent_type(num_buckets, entries_per_bucket, num_blocks,
      emptyKey, hasher, equals),
    hash_table_shared(num_buckets * entries_per_bucket, typename parent_type::HashEntry{emptyKey, 0, 0}),
    bucket_locks_shared(num_buckets, 0),
    data_shared(num_blocks + 1),
    offsets_shared(num_blocks + 2)
  {
    /* prepare the hash table */
    this->hash_table = &hash_table_shared[0];

    /* bucket locks */
    this->bucket_locks = &bucket_locks_shared[0];

    /* allocator -- data*/
    this->alloc.data = &data_shared[0];

    /* allocator -- offsets */
    thrust::sequence(
                  offsets_shared.begin(),
                  offsets_shared.end(),
                  1);
    this->alloc.offsets = &offsets_shared[0];
    this->alloc.mutex = (int*)&offsets_shared[num_blocks];
    offsets_shared[num_blocks] = 0;

    this->alloc.link_head = (int*) &offsets_shared[num_blocks + 1];
    offsets_shared[num_blocks + 1] = num_blocks - 1;
  }

};


}

