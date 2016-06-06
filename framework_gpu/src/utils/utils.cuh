#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>

uint THREADS = 256;
#define BLOCKS(x) max(1, (uint)ceil(x/(float)THREADS))

#define ID threadIdx.x + blockDim.x*blockIdx.x

#define CHECK_ERROR \
{\
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) \
    { \
        fprintf(stderr, "CUDA error: %s(%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define ABS(x) (x ^ (x >> 31)) - (x >> 31)
#define DIST(x1, y1, x2, y2) sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
__forceinline__ __device__ uint MANHATTAN_DIST(int x1, int y1, int x2, int y2) {return ABS(x1-x2)+ABS(y1-y2);}
#endif 

