#ifndef CELLULARSPACE_CUH
#define CELLULARSPACE_CUH

#include "../utils/utils.cuh"
#include "cell.cuh"

template<class C>
__global__ void initializeCellularSpaceKernel(C *cells, uint *quantities, uint *dimensions) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < dimensions[0]*dimensions[1]) {
        C c;
        c.cid = i;
        c.xdim = &dimensions[0];
        c.ydim = &dimensions[1];
        c.quantities = quantities;
        
        //xs[i] = i%xdim;
        //ys[i] = i/xdim;
        quantities[i] = 0;
        cells[i] = c;
    }
}

template<class C>
class CellularSpace{
public:
    CellularSpace(uint xdim, uint ydim) {
        this->xdim = xdim;
        this->ydim = ydim;
        cudaMalloc(&cells, sizeof(C)*xdim*ydim);
        CHECK_ERROR
        cudaMalloc(&dims, sizeof(uint)*2);
        CHECK_ERROR
        cudaMalloc(&quantities, sizeof(uint)*xdim*ydim);
        CHECK_ERROR
    }
    
    ~CellularSpace() {
        cudaFree(cells);
        CHECK_ERROR
        cudaFree(dims);
        CHECK_ERROR
        cudaFree(quantities);
        CHECK_ERROR
    }
    
    void init() {
    	static uint dimensions[2];
    	dimensions[0] = xdim; dimensions[1] = ydim;
    	cudaMemcpy(dims, dimensions, sizeof(uint)*2, cudaMemcpyHostToDevice);
    	CHECK_ERROR
        uint blocks = BLOCKS(xdim*ydim);
        initializeCellularSpaceKernel<<<blocks, THREADS>>>(cells, quantities, dims);
        CHECK_ERROR
    }
    
    uint xdim, ydim;
    uint *dims;
    uint *quantities;
    C *cells;
};

#endif
