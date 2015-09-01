VHashing
====

Reimplementation of Nießner's Voxel Hashing method.

Simply put, to be "neater" and to use thrust classes/functions where possible.


Usage
-----

Refer to `tests/voxelblocks.cu`.

When using the hash table as a function to a kernel call, use HashTableBase:

    __global__
    void kernel(int3 *keys,
        VoxelBlock *values,
        int n,
        vhashing::HashTableBase<int3, VoxelBlock, BlockHasher, BlockEqual> bm) {

This ensures that you don't copy unwanted `thrust::*_vector` structures.

In host code, you should use one of:

1. `HashTable<..., host_memspace>` -- uses `host_vector` (page-locked) in the underlying code
1. `HashTable<..., device_memspace>` -- uses `device_vector` -- therefore accessible from device code.
However, it is not currently accessible from host code.
Therefore `hashTableDevice[key]` will give you a segfault!

You can use copy constructors to convert between memory spaces, e.g.

    auto ht = HashTable<..., host_memspace>(ht_device)

For manipulation of data in the hash table, follow the examples at `tests/filter.cu` and `tests/apply.cu`.

Bulk allocation
-------

Under some circumstances you might find it faster to use `thrust::sort` and
`thrust::unique` to avoid unnecessary conflicts.

If you use `thrust::unique`, you will automatically get a count of the number of elements
you want to insert,
so you can avoid each thread calling `atomicSub(..., 1)` on the allocator.

Use the AllocKeys() functions (TODO: test case, examples)

Caveats
------

1. I used a similar algorithm to Nießner's implementation, which means that **concurrent
inserts and erases are not safe**.
Concurrent **inserts** are safe, as are concurrent **erases**, but not together.
2. If you can tolerate some loss, use `try_insert` and `try_erase`, which gives up
if the lock on the bucket cannot be taken.
If you don't, there might be a small possibility of deadlock -- you have been warned.
(Recall Nießner's voxel hashing -- if some voxel blocks cannot be allocated, we simply let the voxel block be allocated in the next frame)

Very big caveat
======

This package was created from my master thesis, which as all theses go, was a mess
(mainly because I did not divide the HashTable and HashTableBase class, which makes
your reference counting classes do weird things).
So, obviously, my unclean master thesis code is well tested and works, but not this one.
The bulk of the code is the same, but until I also re-implement the TSDF and streaming
and all that stuff
I cannot guarantee that this code _compiles_ when you want to use it.


