#ifndef AGENT_CUH
#define AGENT_CUH

#include "../space/cell.cuh"

class Agent{
public:
    __device__ Agent() : cell(0), dead(0), toDie(0), children(0) {}
    
    __device__ virtual ~Agent(){}
    
    __device__ virtual void init(){}
    
    __device__ virtual void clone(Agent *parent){}
    
    __device__ void move(Cell *cell) {
		nextCell = cell;
	}
    
    __device__ void die(){
		toDie = true;
	}
    
    __device__ void reproduce(uint children) {
		this->children = children;
	}
    
    uint id;
    Cell *cell, *nextCell;
    bool dead, toDie;
    uint children;
};

#endif
