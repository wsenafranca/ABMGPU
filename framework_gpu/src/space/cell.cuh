#ifndef CELL_CUH
#define CELL_CUH

class Cell {
public:
    __device__ Cell() {}
    __device__ virtual ~Cell(){}
    
    __device__ const uint& getX() const {
        return pos.x;
	}
    __device__ const uint& getY() const {
        return pos.y;
	}
    
    __device__ const uint2& getPos() const {
        return pos;
    }
    
    uint cid;
    uint2 pos;
};

#endif

