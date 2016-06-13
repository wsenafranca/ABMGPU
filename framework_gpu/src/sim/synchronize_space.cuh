#ifndef SYNCHRONIZESPACE_CUH
#define SYNCHRONIZESPACE_CUH

template<class C>
class SynchronizeSpaceAction : public Action {
public:
    SynchronizeSpaceAction(CellularSpace<C> *cellSpace) : cs(cellSpace) {}
    void action(cudaStream_t &stream) {
        uint blocks = BLOCKS(cs->size);
        cudaMemcpyAsync(cs->getPastDevice(), cs->getCellsDevice(), sizeof(C)*cs->size, cudaMemcpyDeviceToDevice, stream);
    }
    CellularSpace<C> *cs;
};

#endif


