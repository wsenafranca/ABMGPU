#ifndef ENVIRONMENT_CUH
#define ENVIRONMENT_CUH

#include "../utils/utils.cuh"

class Environment{
public:
    Environment() : size(0), d_mem(0) {}
    
    virtual ~Environment() {
        cudaFree(d_mem);
        CHECK_ERROR
    }
    
    bool init() {
        cudaError_t err = cudaMalloc(&d_mem, size);
        if(err != cudaSuccess) {
            fprintf(stderr, "%s\n", cudaGetErrorString(err));
            return false;
        }
        return true;
    }
    
    uint alloc(uint bytes) {
        uint index = size;
        size+=bytes;
        return index;
    }
    
    unsigned char* getGlobalMemory() {
        return d_mem;
    }
    
    static Environment* getEnvironment() {
        if(!environment) environment = new Environment();
        return environment;
    }
    
private:
    uint size;
    unsigned char *d_mem;
    static Environment* environment;
};

Environment *Environment::environment = 0;

#endif
