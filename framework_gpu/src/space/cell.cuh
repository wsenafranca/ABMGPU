#ifndef CELL_CUH
#define CELL_CUH

class Cell {
public:
    __device__ Cell(){}
    __device__ virtual ~Cell(){}
    __device__ uint getQuantity() const {
		return quantities[cid];
	}
    __device__ uint getX() const {
		return cid%(*xdim);
	}
    __device__ uint getY() const {
		return cid/(*xdim);
	}

    uint cid;
    uint *quantities;
    uint *xdim, *ydim;
};

#endif
