#include "impl/lockset.h"
#include <safecall.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>

using namespace std;
using namespace vhashing;
using vhashing::detail::Lock;
using vhashing::detail::LockSet;

__global__
void kernel(int *table, Lock *locks, int tableSize, int *failed)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i=idx; i<tableSize; i++) {
		bool done = false;

		while (!done) {
			LockSet<5> lockset;
			if (lockset.TryLock(locks[i])) {
				assert(lockset.TryLock(locks[i]));
				assert(lockset.TryLock(locks[i]));

				table[i]++;
				done = true;
			}
		}
	}
}


int main() {
	int *table;
	Lock *locks;
	int *failed;
	int tableSize = 5000;

	cudaSafeCall(cudaMalloc(&table, sizeof(int) * tableSize));
	cudaSafeCall(cudaMalloc(&locks, sizeof(Lock) * tableSize));
	cudaSafeCall(cudaMalloc(&failed, sizeof(int) * tableSize));

	cudaSafeCall(cudaMemset(table, 0, sizeof(int) * tableSize));
	cudaSafeCall(cudaMemset(locks, 0, sizeof(Lock) * tableSize));
	cudaSafeCall(cudaMemset(failed, 0, sizeof(int) * tableSize));

	cudaSafeCall(cudaDeviceSynchronize());
	fprintf(stderr, "post-memset\n");
	{
    int numjobs = tableSize;
		int threadsPerBlock = 256;
		int numblocks = (numjobs + threadsPerBlock - 1) / threadsPerBlock;

		fprintf(stderr, "%d %d\n", numblocks, threadsPerBlock);

		kernel<<<numblocks, threadsPerBlock>>>(table, locks, tableSize, failed);
	}
	cudaSafeCall(cudaDeviceSynchronize());
	fprintf(stderr, "post-syncrhonize\n");
	
	{
		int *result = (int*) malloc(sizeof(int) * tableSize);
		cudaSafeCall(cudaMemcpy(result, table, sizeof(int) * tableSize, cudaMemcpyDeviceToHost));

		for (int i=0; i<tableSize; i++) {
			if (result[i] == i+1) {
			}
			else {
				fprintf(stderr, "[%d] is wrong == (%d)\n", i+1, result[i]);
			}
		}
	}

	{
		int *hfailed = (int*) malloc(sizeof(int) * tableSize);
		cudaSafeCall(cudaMemcpy(hfailed, failed, sizeof(int) * tableSize,
					cudaMemcpyDeviceToHost));

		for (int i=0; i<tableSize; i++) {
			if (hfailed[i]) {
				printf("Failure at position %d\n", i);
			}
		}
	}

	cudaFree(table);
	cudaFree(locks);
}


