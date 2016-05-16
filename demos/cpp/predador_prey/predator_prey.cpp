#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>

#include <set>
#include <list>

int POP_SIZE;
#define ENERGY 40
#define GROWTH 4

size_t XDIM;
size_t YDIM;

#define ITERATION 1000

#define PASTURE 0
#define SOIL 1
#define PREDATOR 0
#define PREY 1
                                                                                                        
#define random() (rand()/(float)RAND_MAX)

size_t predatorPop = 0;
size_t preyPop = 0;

class Agent;
class Society;

class Cell{
public:
    Cell() : cover(0), count(0), empty(1), target(0){}
    
    std::set<Agent*> agents;
    
    size_t x;
    size_t y;
    int cover;
    int count;
    bool empty;
    Agent *target;
};

class CellularSpace{
public:
    CellularSpace(size_t xDim, size_t yDim):xdim(xDim), ydim(yDim) {
        cells  = new Cell[xdim*ydim];
        for(size_t i = 0, k = 0; i < ydim; i++) {
            for(size_t j = 0; j < xdim; j++, k++) {
                cells[k].x = j;
                cells[k].y = i;
            }
        }
    }
    
    ~CellularSpace() {
        delete [] cells;
    }
    
    size_t xdim, ydim;
    Cell *cells;
};

class Agent{
public:
    Agent() : cell(NULL), nextCell(NULL), numTargets(0), energy(ENERGY), nextEnergy(ENERGY), type(0), toDie(false), dead(false), child(false), toMove(false){}
    
    void move(Cell *cell) {
        toMove = true;
        nextCell = cell;
    }
    
    void reproduce() {
        child = true;
    }
    
    void die() {
        toDie=true;
    }
    
    void eat(Agent *other) {
        if(numTargets < 100) {
            targets[numTargets++] = other;
            other->predator = this;
        }        
    }
    
    void eat() {
        cell->target = cell->target ? cell->target : this;
    }
    
    Agent* commitReproduce() {
        Agent *ag = new Agent();
        ag->type = type;
        child = false;        
        return ag;
    }
    
    bool commitMove() {
        if(!dead && toMove) {
            Cell *newCell = nextCell;
            Cell *oldCell = cell;
            cell = newCell;
            newCell->agents.insert(this);
            if(oldCell) oldCell->agents.erase(this);
            toMove = false;
            nextCell = 0;
        }
        
        return true;
    }
    
    void commitEat() {
        if(type==PREDATOR) {
            for(int i = 0; i < numTargets; i++) {
                Agent *prey = targets[i];
                if(prey->predator == this) {
                    nextEnergy += prey->energy/2;
                    prey->die();
                    break;
                }
            }
            numTargets = 0;
        }
        else {
            if(cell->target == this && cell->cover == PASTURE) {
                cell->cover = SOIL;
                nextEnergy = nextEnergy+5;
            }
        }
    }
    
    Cell *cell, *nextCell;
    Agent* targets[100];
    int numTargets;
    Agent* predator;
    
    //=========================================
    int energy, nextEnergy;
    int type;
    bool toDie;
    bool dead;
    bool child;
    bool toMove;
};

class Society{
public:
    Society(size_t quantity, size_t capacity) {
        this->capacity = capacity;
        for(size_t i = 0; i < quantity; i++) {
            Agent *ag = new Agent();
            ag->type = i < quantity * 0.25f ? PREDATOR : PREY;
            agents.insert(ag);
            if(ag->type == PREDATOR) predatorPop++;
            else preyPop++;
        }
    }
    ~Society() {
        std::set<Agent*>::iterator it;
        for(it = agents.begin(); it != agents.end(); it++)
            delete *it;
    }
    size_t getQuantity() const {
        return agents.size();
    }
    
    std::set<Agent*> agents;
    size_t capacity;
};

void regrowth(Cell *cell) {
    if(cell->cover == SOIL) {
        cell->count++;
        if(cell->count >= GROWTH){
            cell->cover = PASTURE;
            cell->count = 0;
            cell->target = NULL;
        }
    }
}

void placement(Society *soc, CellularSpace *cs) {
    std::set<Agent*>::iterator it;
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        size_t cellId = rand()%(cs->ydim*cs->xdim);
        ag->move(&cs->cells[cellId]);
        ag->commitMove();
    }
}

void commonActions(Agent *ag, CellularSpace *cs) {   
    ag->nextEnergy--;
    if(ag->energy <= 0) {
        ag->die();
        return;
    }
    
    Cell *cell = ag->cell;
    
    static const int neighs[][2] = {{0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {0, 1}, {0, -1}};
    int pos = rand()%8;
    int rx = neighs[pos][0];
    int ry = neighs[pos][1];
    
    size_t newX = cell->x + rx;
    size_t newY = cell->y + ry;

    if(newX < cs->xdim && newY < cs->ydim) {
        ag->move(&cs->cells[newY*cs->xdim+newX]);
    }
    
    if((ag->type == PREY && ag->energy >= 50) || (ag->type == PREDATOR && ag->energy >= 50)) {
        ag->nextEnergy /= 2;
        ag->reproduce();
    }
    
    if(ag->type == PREY)
    	ag->eat();
}

void hunt(Agent *ag, Society *soc, CellularSpace *cs) {
    if(ag->dead) return;
    std::set<Agent*>::iterator it;
    for(it = ag->cell->agents.begin(); it != ag->cell->agents.end(); it++) {
        Agent *other = *it;
        if(!other->dead && other != ag)
            if(other->type == PREY || random() < 0.1)
                ag->eat(other);
    }
}

void action(Agent *ag, CellularSpace *cs) {
	commonActions(ag, cs);
	ag->commitEat();
}

void synchronize(Society *soc, CellularSpace *cs) {
    std::list<Agent*> deads;
    std::list<Agent*> children;
    std::set<Agent*>::iterator it;
    int nchildren = 0;
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        if(ag->toDie) {
            ag->dead=true;
            deads.push_back(ag);
        }
        else if(ag->child && soc->getQuantity()+nchildren < soc->capacity-1) {
            Agent *child = ag->commitReproduce();
            size_t cid = rand()%(cs->xdim*cs->ydim);
            //child->move(&cs->cells[cid]);
            child->move(ag->cell);
            //child->commitMove();
            children.push_back(child);
            nchildren++;
        }
    }
    //printf("%ld %ld\n", deads.size(), children.size());
    {
        std::list<Agent*>::iterator it;
        for(it = deads.begin(); it != deads.end(); it++) {
            Agent *ag = *it;
            if(ag->cell) {
                ag->cell->agents.erase(ag);
            }
            
            if(ag->type == PREDATOR) predatorPop--;
            else preyPop--;
            
            soc->agents.erase(ag);
            delete ag;
        }
    }
    {
        std::list<Agent*>::iterator it;
        for(it = children.begin(); it != children.end(); it++) {
            Agent *child = *it;
            soc->agents.insert(child);
            if(child->type == PREDATOR) predatorPop++;
            else preyPop++;
        }
    }
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        ag->commitMove();
    }
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        ag->energy = ag->nextEnergy;
    }    
}

void execute(CellularSpace *cs) {
    for(size_t i = 0; i < XDIM*YDIM; i++)
        regrowth(&cs->cells[i]);
}

void execute(Society *soc, CellularSpace *cs) {
    std::set<Agent*>::iterator it;    
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        if(!ag->dead && ag->type == PREDATOR) hunt(ag, soc, cs);
    }
    for(it = soc->agents.begin(); it != soc->agents.end(); it++) {
        Agent *ag = *it;
        if(!ag->dead) action(ag, cs);
    }
    
}

int saveBMP(CellularSpace *cs, unsigned int width, unsigned int height, const char *filename) {
    unsigned int filesize = 54 + 3*width*height;
    
    unsigned char *img = (unsigned char*)malloc(sizeof(unsigned char)*width*height*3);
    
    for(unsigned int i = 0; i < height*width; i++) {
        unsigned char r = 0, g = 0, b = 0;
        if(!cs->cells[i].agents.empty()) {
            if((*cs->cells[i].agents.begin())->type==PREDATOR) {
                r = 255;
                g = 0;
                b = 0;
            }
            else {
                r = 0;
                g = 0;
                b = 255;
            }
        }
        else {
            if(cs->cells[i].cover == SOIL) {
                r = g = b = 0;
            }
            else{
                r = b = 0;
                g = 255;
            }
        }
        
        img[i*3+2] = r;
        img[i*3+1] = g;
        img[i*3+0] = b;
    }
    
    unsigned char bmpfileheader[14] = {'B','M', 0,0,0,0, 0,0, 0,0, 54,0,0,0};
    unsigned char bmpinfoheader[40] = {40,0,0,0, 0,0,0,0, 0,0,0,0, 1,0, 24,0};
    unsigned char bmppad[3] = {0,0,0};
    
    bmpfileheader[ 2] = (unsigned char)(filesize    );
    bmpfileheader[ 3] = (unsigned char)(filesize>> 8);
    bmpfileheader[ 4] = (unsigned char)(filesize>>16);
    bmpfileheader[ 5] = (unsigned char)(filesize>>24);

    bmpinfoheader[ 4] = (unsigned char)(width);
    bmpinfoheader[ 5] = (unsigned char)(width>> 8);
    bmpinfoheader[ 6] = (unsigned char)(width>>16);
    bmpinfoheader[ 7] = (unsigned char)(width>>24);
    bmpinfoheader[ 8] = (unsigned char)(height);
    bmpinfoheader[ 9] = (unsigned char)(height>> 8);
    bmpinfoheader[10] = (unsigned char)(height>>16);
    bmpinfoheader[11] = (unsigned char)(height>>24);
    
    FILE *f = fopen(filename,"wb");
    fwrite(bmpfileheader,1,14,f);
    fwrite(bmpinfoheader,1,40,f);
    for(unsigned i=0; i<height; i++) {
        fwrite(img+(width*(height-i-1)*3),3,width,f);
        fwrite(bmppad,1,(4-(width*3)%4)%4,f);
    }
    fclose(f);
    
    free(img);
    return 0;
}

void runModel(long seed) {
    srand(seed);
    Society pop(POP_SIZE, XDIM*YDIM);
    CellularSpace cs(XDIM, YDIM);
    placement(&pop, &cs);
    //saveBMP(&cs, XDIM, YDIM, "img0.bmp");
    //printf("%ldx%ld(%ld)\n", XDIM, YDIM, XDIM*YDIM);
    //printf("time total predators preys\n");
    //float popMean = 0;
    for(int i = 1; i <= ITERATION; i++) {
        //printf("%d %ld %ld %ld\n", i, predatorPop+preyPop, predatorPop, preyPop);
        synchronize(&pop, &cs);
        execute(&pop, &cs);
        execute(&cs);
        //popMean += pop.getQuantity();
    }
    //printf("popMean: %f\n", popMean/ITERATION);
    //saveBMP(&cs, XDIM, YDIM, "img1.bmp");
}

int main() {
    size_t sizes[] = {128, 256, 512, 768, 960};
    //size_t sizes[] = {32, 48, 64, 96, 128, 192, 256, 388, 512, 1024, 1280, 1600, 2240, 2480, 2560};
    //system("date");
    long t;
    FILE *f = fopen("preys_CPU.txt", "w");
    for(int i = 0; i < 5; i++) {
        XDIM = sizes[i];
        YDIM = sizes[i];
        POP_SIZE = truncf(XDIM*YDIM*0.1f);
        fprintf(f, "Test %d: Dim: %ld (%lf)\n", i+1, XDIM, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
        printf("Test %d: Dim: %ld (%lf)\n", i+1, XDIM, (double)(double)POP_SIZE/(double)(XDIM*YDIM));
        for(int j = 0; j < 5; j++) {
            t = clock();
            runModel(j*10);
            t = clock()-t;
            printf("elapsed time %lf secs\n", t/(double) CLOCKS_PER_SEC);
            fprintf(f, "elapsed time %lf secs\n", t/(double) CLOCKS_PER_SEC);
        }
    }
    
    //system("date");
    fclose(f);
	return 0;
    
}

