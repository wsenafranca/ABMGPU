#ifndef NEIGHBORHOOD_CUH
#define NEIGHBORHOOD_CUH

#include "../space/cellularspace.cuh"
#include "../agents/society.cuh"

template<class A, class C>
class Neighborhood {
public:
    Neighborhood(Society<A> *soc, CellularSpace<C> *cs, uint n, uint m) {
        this->n = n;
        this->m = m;
        cudaMalloc(&neighborhood, sizeof(A*)*soc->capacity);
        CHECK_ERROR
        cudaMalloc(&inhabited, sizeof(C*)*soc->capacity);
        CHECK_ERROR
        cudaMalloc(&size, sizeof(uint));
        CHECK_ERROR
        cudaMalloc(&offset, sizeof(uint)*cs->xdim*cs->ydim);
        CHECK_ERROR;
        cudaMalloc(&registred, sizeof(uint)*cs->xdim*cs->ydim);
        CHECK_ERROR;
        cudaMalloc(&pos, sizeof(uint)*cs->xdim*cs->ydim);
        CHECK_ERROR;
        
        cudaMemset(registred, 0, sizeof(uint)*cs->xdim*cs->ydim);
        cudaMemset(pos, 0, sizeof(uint)*cs->xdim*cs->ydim);
    }
    ~Neighborhood() {
        cudaFree(neighborhood);
        CHECK_ERROR
        cudaFree(inhabited);
        CHECK_ERROR
        cudaFree(size);
        CHECK_ERROR
        cudaFree(offset);
        CHECK_ERROR
        cudaFree(registred);
        CHECK_ERROR
        cudaFree(pos);
        CHECK_ERROR
    }
    A **neighborhood;
    C **inhabited;
    uint *pos;
    uint *size;
    uint *offset;
    uint *registred;
    uint n, m;
};

#endif
