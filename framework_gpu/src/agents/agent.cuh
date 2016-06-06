#ifndef AGENT_CUH
#define AGENT_CUH

#include "../space/cell.cuh"

class Agent{
public:
    __device__ Agent() : cell(0), dead(0), children(0){}
    
    __device__ virtual ~Agent(){}
    
    __device__ virtual void init(){}
    
    __device__ virtual void clone(Agent *parent){}
    
    __device__ void move(Cell *cell) {
		this->cell = cell;
	}
    
    __device__ void die() {
        dead = true;
	}
    
    __device__ bool isDead() const{
        return dead;
    }
    
    __device__ void reproduce(uint children) {
        this->children = children;
    }
    
    //uint id;
    Cell *cell;
    bool dead;
    uint children;
};

#endif

