#ifndef NEIGHBORHOOD_CUH
#define NEIGHBORHOOD_CUH

#include "../space/cellularspace.cuh"
#include "../agents/society.cuh"

template<class A, class C>
class Neighborhood {
public:
    Neighborhood(Society<A> *soc, CellularSpace<C> *cs, uint n, uint m) : cellularSpace(cs), society(soc) {
        this->n = n;
        this->m = m;
        
        uint globalMemory = 0;
        globalMemory += sizeof(A*)*soc->capacity;           // alloc neighborhood
        globalMemory += sizeof(C*)*soc->capacity;           // alloc inhabited
        globalMemory += sizeof(uint);                       // alloc size
        globalMemory += sizeof(uint)*cs->xdim*cs->ydim;     // offset
        globalMemory += sizeof(uint)*cs->xdim*cs->ydim;     // registred
        globalMemory += sizeof(uint)*cs->xdim*cs->ydim;     // pos
        
        neighborhoodIndex = 0;
        inhabitedIndex = neighborhoodIndex + sizeof(A*)*soc->capacity;
        sizeIndex = inhabitedIndex+sizeof(C*)*soc->capacity;
        offsetIndex = sizeIndex + sizeof(uint);
        registredIndex = offsetIndex + sizeof(uint)*cs->xdim*cs->ydim;
        posIndex = registredIndex + sizeof(uint)*cs->xdim*cs->ydim;
        
        index = Environment::getEnvironment()->alloc(globalMemory);
    }
    
    ~Neighborhood() {
    }
    
    A** getNeighborhoodDevice() {
        return (A**)(Environment::getEnvironment()->getGlobalMemory()+index+neighborhoodIndex);
    }
    
    C** getInhabitedDevice() {
        return (C**)(Environment::getEnvironment()->getGlobalMemory()+index+inhabitedIndex);
    }
    
    uint* getSizeDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+sizeIndex);
    }
    
    uint* getOffsetDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+offsetIndex);
    }
    
    uint* getRegistredDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+registredIndex);
    }
    
    uint* getPosDevice() {
        return (uint*)(Environment::getEnvironment()->getGlobalMemory()+index+posIndex);
    }
    
    void init() {
        cudaMemset(getRegistredDevice(), 0, sizeof(uint)*cellularSpace->xdim*cellularSpace->ydim);
        cudaMemset(getPosDevice(), 0, sizeof(uint)*cellularSpace->xdim*cellularSpace->ydim);
    }
    
    uint neighborhoodIndex;
    uint inhabitedIndex;
    uint sizeIndex;
    uint offsetIndex;
    uint registredIndex;
    uint posIndex;
    uint index;
    uint n, m;
    CellularSpace<C> *cellularSpace;
    Society<A> *society;
};

#endif
