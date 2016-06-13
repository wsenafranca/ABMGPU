#include <simgpu.cuh>

class MyAgent : public Agent{
public:
    __device__ void init() {
        x = cuRand(ID, 0, 10);
    }
    __device__ void add() {
        x++;
    }
    int x;
};

__device__ void addX(Agent *agent) {
    MyAgent* ag = (MyAgent*)agent;
    ag->add();
    printf("%u: %d\n", ID, ag->x);
}

template<SocialChange change, class A>
class ExecuteAction : public Action{
public:
    ExecuteAction(Society<A> *society) :soc(society) {}
    
    void action(cudaStream_t &stream) {
        uint blocks = BLOCKS(soc->size);
        executeKernel<change, A><<<blocks, THREADS, 0, stream>>>(soc->getAgentsDevice(), soc->size);
        CHECK_ERROR;
    }
    
    Society<A> *soc;
};


int main() {
    Environment *env = new Environment();
    env->setTimer(new Timer(10));
    
    for(int i = 0; i < 10; i++) {
        Society<MyAgent> *soc = env->createSociety<MyAgent>(10);
        env->getTimer()->createEvent(1, 1, 1, new ExecuteAction<addX, MyAgent>(soc));
    }
    
    env->execute(10);
    
    delete env;
    return 0;
}


