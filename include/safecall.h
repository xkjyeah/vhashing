#ifndef DSVH_SAFECALL_H
#define DSVH_SAFECALL_H

#include <cuda_runtime.h>

#define cudaSafeCall( EXPR ) __cudaSafeCall(EXPR, __FILE__, __LINE__)
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)
#define cudaCheckKernel() __cudaCheckKernel(__FILE__, __LINE__)
#define cudaCheckKernelStream(stream) __cudaCheckKernelStream(stream, __FILE__, __LINE__)
#define cudaCheckMemory() __cudaCheckMemory(__FILE__, __LINE__)

void __cudaSafeCall( cudaError err, const char *file, const int line );
void __cudaCheckKernel( const char *file, const int line );
void __cudaCheckKernelStream( cudaStream_t stream, const char *file, const int line );
void __cudaCheckError( const char *file, const int line ); 
void __cudaCheckMemory( const char *file, const int line ); 

#endif
