#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>

#define GPU __device__ __forceinline__

static int CURRENT_DEVICE = 0;
void setCurrentDevice() {
    cudaSetDevice(CURRENT_DEVICE);
}
uint THREADS = 256;
#define BLOCKS(x) max(1, (uint)ceil(x/(float)THREADS))

#define ID threadIdx.x + blockDim.x*blockIdx.x

// needed to orders the pre-execution (time=0) on scheduler
#define COLLECTION_PRIORITY -4
#define PLACEMENT_PRIORITY -3
#define SYNCHRONIZE_PRIORITY -2
#define INDEXING_PRIORITY -1

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

