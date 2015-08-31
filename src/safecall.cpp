#include "safecall.h"
#include <iostream>
#include <glog/logging.h>
#include <cstdlib>
#include <cstdio>

using std::fprintf;

void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	if ( cudaSuccess != err )
	{
    {
			using namespace std;
			size_t free, total;
			cudaMemGetInfo(&free, &total);

			LOG(WARNING) << free << endl;
			LOG(WARNING) << total << endl;
		}
		LOG(ERROR) << "cudaSafeCall() failed at "
      << file << ":"
      << line << " : "
      << cudaGetErrorString( err );
		throw "CUDA Error";
	}
	 
	return;
}
void __cudaCheckError( const char *file, const int line )
{
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		LOG(ERROR) << "cudaCheckError() failed at "
      << file << ":"
      << line << " : "
      << cudaGetErrorString( err );
		throw "CUDA Error";
	}
 return;
} 

void __cudaCheckMemory(const char *file, const int line)
{
	size_t free, total;

	cudaMemGetInfo(&free, &total);
  VLOG(0) << "CUDA Memory at "
    << file << ":"
    << line << " : "
    << "free(" << free << ")"
    << "total(" << total << ")";
}

void __cudaCheckKernel(const char *file, const int line)
{
	__cudaSafeCall(cudaDeviceSynchronize(), file, line);
	__cudaCheckError(file, line);
}

void __cudaCheckKernelStream(cudaStream_t stream, const char *file, const int line)
{
	__cudaSafeCall(cudaStreamSynchronize(stream), file, line);
	__cudaCheckError(file, line);
}
