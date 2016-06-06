#include<simgpu.cuh>

enum Type{
    PREDATOR,PREY
};

const uint ENERGY_INITIAL = 40;
const uint ENERGY_GAIN = 4;
const uint ENERGY_LOST = 1;
const uint REPRODUCTION_THRESHOLD = 20;
const float REPRODUCTION_PROBABILITY 0.3f;
const uint CELL_RESTART = 4;


class MyAgent : public Agent{
public:
    __device__ void init() {
        type = cuRandom(ID) > 0.5f ? PREDATOR : PREY;
        energy = ENERGY_INITIAL;
    }
    __device__ void clone(Agent *p) {
        MyAgent *parent = (MyAgent*)p;
        type = parent->type;
        energy /= 2;
    }
    
    int energy;
    Type type;
    MyAgent *target;
};

class MyCell : public Cell{
public:
    uint count;
    MyAgent *target;
}

__global__ void move(MyAgent *agents, uint size, MyCell *cells, uint xdim, uint ydim) {
    uint id = ID;
    if(id < size) {
        MyCell *cell = (MyCell*)agents[id].cell;
        uint2 pos = cell->getPos();
        int x = pos.x;
        int y = pos.y;
        int nx = cuRand(id, -1, 1)+x;
        int ny = cuRand(id, -1, 1)+y;
        uint newcid = y*xdim + x;
        if(newcid < xdim*ydim) {
            agents[id].move(&cells[newcid]);
        }
        agents[id].energy -= ENERGY_LOST;
    }
}

__device__ void growFood(MyCell *cells, uint size) {
    uint id = ID;
    if(id < size) {
        uint c = cells[id].count;
        cells[id].count = c > 0 ? c-1 : 0;
    }
}

__device__ void tryEatPredator(MyAgent *ag, MyCell *cell, MyAgent *current, MyAgent *past, uint *neighborhood) {
    uint pos = cell->getPos();
    uint begin = tex2D(beginsRef, pos.x, pos.y);
    uint end = tex2D(beginsRef, pos.x, pos.y);
    uint *p = &neighborhood[begin];
    uint *e = &neighborhood[end];
    while(p != e) {
        MyAgent *other = past[*p];
        if(other->type == PREY) { // predator targets prey
            ag->target = other;
            other->target = ag;
            break;
        }
        p++;
    }
}

__device__ void tryEatPrey(MyAgent *ag, MyCell *cell) {
    cell->target = ag;
}

__global__ void tryEat(MyAgent *agents, MyAgent *past, uint size, uint *neighborhood) {
    uint id = ID;
    if(id < size) {
        MyCell *cell = (MyCell*)agents[id].cell;
        MyAgent *ag = &agents[id];
        if(ag->type == PREDATOR) tryEatPredator(ag, cell, agents, past, neighborhood);
        else tryEatPrey(ag, cell);
    }
}

__global__ void eat(MyAgent *agents, uint size) {
    uint id = ID;
    if(id < size) {
        MyAgent *ag = &agents[id];
        if(ag->type == PREDATOR) {
            if(ag->target && ag->target == ag->target->target) {
                MyAgent *prey = ag->target;
                ag->energy += ENERGY_GAIN;
                prey->die();
            }
        }
        else {
            MyCell *cell = (MyCell*)agents[id].cell;
            if(cell->target == ag) {
                cell->count = CELL_RESTART;
                ag->energy += ENERGY_GAIN;
            }
        }
    }
}

__device__ void tryReproduce(MyAgent *ag) {
    if(ag->energy > REPRODUCTION_THRESHOLD) {
        if(cuRandom(ID) < REPRODUCTION_PROBABILITY) {
            ag->energy /= 2;
        }
    }
}

__global__ void act(Agent *agents, uint size) {
    uint id = ID;
    if(id < size) {
        tryReproduce(&agents[id]);
    }
}

int main() {
    Society<MyAgent> soc(100000, 1000000);
    CellularSpace<MyCell> cs(1000, 1000);
    Neighborhood<MyAgent, MyCell> nb(&soc, &cs);
    
    Environment::getEnvironment()->init();
    
    soc.init();
    cs.init();
    nb.init();
    placement(&soc, &cs);
    
    uint sblocks = BLOCKS(cs.xdim*cs.ydim);
    for(uint i = 0; i < 1000; i++) {
        synchronize(&soc, &cs, &nb);
        uint blocks = BLOCKS(soc.size);
        move<<<blocks, THREADS>>>(soc.getAgentsDevice(), soc.size, cs.getCellsDevice(), cs.xdim, cs.ydim());
        growFood<<<sblocks, THREADS>>>(cs.getCellsDevice(), cs.xdim, cs.ydim);
    }
    
    delete Environment::getEnvironment(); // temp
    return 0;
}

