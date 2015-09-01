#include "vhashing.h"
#include <unordered_map>
#include <utility>
#include <random>
#include <cuda_runtime.h>

using namespace std;

struct Data {
	int r[1026];
	bool operator==(const Data & other ){
		for (int i=0;i<1026;i++){
			if (other.r[i] != r[i])
				return false;
		}
		return true;
	}
};
struct hasher {
	__device__ __host__
	size_t operator() (int3 x) const { return x.x ^ x.y ^ x.z; }
};
struct kequal {
	__device__ __host__
	bool operator()(int3 x, int3 y) const {
	return x.x == y.x && x.y == y.y && x.z == y.z;
	}
};

typedef vhashing::HashTable<int3, Data, hasher, kequal, vhashing::device_memspace> BM;
typedef vhashing::HashTableBase<int3, Data, hasher, kequal> BMBase;
typedef vhashing::HashTable<int3, Data, hasher, kequal, vhashing::std_memspace> BMH;

__global__
void kernel(int3 *keys, Data *values, int n, int tasks, BMBase bm) {
	int base = blockDim.x * blockIdx.x  +  threadIdx.x;

	for (int i=0; i<tasks; i++) {
		int idx = base * tasks + i;

		if (idx >= n)
			return;

		bm[keys[idx]] = values[idx];
	}
}

/**
  Creates an unordered_map (mblocks), a local copy (hblocks).

  Inserts the same values into mblocks and hblocks.
  
  Uses a kernel to copy values from hblocks to a device hashtable (blocks)

  Uses the cross-memory space copy constructor to copy into bmh

  Check that bmh values == mblocks values
  **/
int main() {

	BM blocks(10000, 6, 19997, make_int3(-1,-1,-1), hasher(), kequal());
	BMH hblocks(10000, 6, 19997, make_int3(-1,-1,-1), hasher(), kequal());
	std::unordered_map<int3, Data, hasher, kequal> mblocks(10,hasher(), kequal());

	vector< int3 > keys;
	vector< Data > values;

	default_random_engine dre;
	for (int i=0; i<10000; i++) {
		int3 k = make_int3( dre(), dre(), dre() );
		Data d;

		for (int j=0;j<1026;j++) {
			d.r[j] = dre();
		}
		
		values.push_back(d);
		keys.push_back(k);
		mblocks[k] = d;
		hblocks[k] = d;
	}

	printf("Link head at %d\n", *hblocks.alloc.link_head);

	printf("Generated values\n");

	// insert into blockmap
	{
		int3 *dkeys;
		Data *dvalues;

		cudaSafeCall(cudaMalloc(&dkeys, sizeof(int3) * keys.size()));
		cudaSafeCall(cudaMalloc(&dvalues, sizeof(Data) * keys.size()));

		cudaSafeCall(cudaMemcpy(dkeys, &keys[0], sizeof(int3) * keys.size(), cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMemcpy(dvalues, &values[0], sizeof(Data) * keys.size(), cudaMemcpyHostToDevice));

		printf("Running kernel\n");

//		kernel<<<keys.size() / 256 + 1,256>>>(dkeys, dvalues, keys.size(), blocks);
		kernel<<<2,256>>>(dkeys, dvalues, keys.size(), keys.size() / 512 + 1, blocks);

		cudaSafeCall(cudaDeviceSynchronize());
	}

	printf("Copying back\n");


	// stream in
	BMH bmh(blocks);

	// check
	for (auto &&it : mblocks) {
		printf(">> [%d %d %d] = %p / %p / %p \n", it.first.x, it.first.y, it.first.z, bmh[it.first].r, it.second.r, hblocks[it.first].r);

		printf("\t\tpointer: %p\n", bmh.find(it.first)->block_index);
		printf("\t\tpointer: %p\n", hblocks.find(it.first)->block_index);

		if (!(bmh[it.first] == it.second)) {
			printf("FAILED\n");
			throw 1;
		}
	}

	// reverse check
	

	return 0;

}
