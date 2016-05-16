#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdio>

#define THREADS 128
#define BLOCKS(x) max(1, (uint)ceil(x/(float)THREADS))

#define CHECK_ERROR \
{\
    cudaError_t error = cudaGetLastError(); \
    if(error != cudaSuccess) \
    { \
        char message[256]; \
        sprintf(message, "CUDA error: %s(%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        printf("Error: %s\n", message); exit(1);\
    } \
}

#define DIST(x1, y1, x2, y2) sqrtf((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

#endif 
