#ifndef SYNCHRONIZESPACE_CUH
#define SYNCHRONIZESPACE_CUH

template<class A, class C>
__global__ void findInhabitedKernel(A *agents, uint size, C **inhabited, uint *registred, uint *pos) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        A *ag = &agents[i];
        if(ag->dead) return;
        
        C *cell = (C*)ag->cell;
        if(atomicCAS(&registred[cell->cid], 0, 1)==0) {
            uint p = atomicAdd(pos, 1);
            inhabited[p] = cell;
        }
    }
}

template<class C>
__global__ void calcOffsetKernel(C **cells, uint *offset, uint *quantities, uint len) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < len) {
        uint p = 0;
        for(uint j = 0; j < i; j++) {
        	p+=quantities[cells[j]->cid];
        }
        offset[cells[i]->cid] = p;
    }
}

template<class C>
__global__ void prescan(C **cells, uint *offsets, uint *quantities, uint *buffer, uint len) {
	extern __shared__ uint s_data[];
	uint tid = threadIdx.x;
    uint i = threadIdx.x+blockDim.x*blockIdx.x;
    uint offset = 1;
    s_data[tid] = (i < len) ? quantities[cells[i]->cid] : 0;
    for(uint d = blockDim.x>>1; d > 0; d>>=1) {
        __syncthreads();
        if(tid < d) {
            uint ai = offset*(2*tid+1)-1;
            uint bi = offset*(2*tid+2)-1;
            s_data[bi]+=s_data[ai];
        }
        offset*=2;
    }
    
    if(tid == 0) {
        if(buffer) buffer[blockIdx.x] = s_data[blockDim.x-1];
        s_data[blockDim.x-1] = 0; 
    }
    
    for(uint d = 1; d < blockDim.x; d*=2) {
        offset>>=1;
        __syncthreads();
        if(tid < d) {
            uint ai = offset*(2*tid+1)-1;
            uint bi = offset*(2*tid+2)-1;
            uint t = s_data[ai];
            s_data[ai]=s_data[bi];
            s_data[bi]+=t;
        }
    }
    __syncthreads();
    if(i < len)
    	offsets[cells[i]->cid] = s_data[tid];
}

template<class C>
__global__ void postscan(C **cells, uint *buffer, uint *offset, uint len) {
	__shared__ uint buf;
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < len) 
        buf = buffer[blockIdx.x];
        
    __syncthreads();
    
    if(i < len) 
        offset[cells[i]->cid]+=buf;
}

template<class C>
void scan(C **cells, uint *offsets, uint *quantities, uint len) {
	uint size = next2k(len);
    uint blocks = BLOCKS(size);
    
	uint *d_buffer;
    cudaMalloc(&d_buffer, sizeof(uint)*blocks);
    
    uint threads = min(THREADS, size);
    prescan<<<blocks, threads, sizeof(uint)*(size/blocks)>>>(cells, offsets, quantities, d_buffer, len);
    CHECK_ERROR
    recursiveScan(d_buffer, d_buffer, blocks);
    CHECK_ERROR
    postscan<<<blocks, threads, sizeof(uint)>>>(cells, d_buffer, offsets, len);
    CHECK_ERROR
    
    cudaFree(d_buffer);
}

template<class A>
__global__ void sortAgentsKernel(A *agents, uint size, A **neighborhood, uint *offset, uint *pos) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        A *ag = &agents[i];
        if(!ag->dead) {
            Cell *cell = ag->cell;
            uint cid = cell->cid;
            uint init = offset[cid];
            uint p = atomicAdd(&(pos[cid]), 1);
            neighborhood[init+p] = ag;
        }
    }
}

template<class C>
__global__ void clearInfoKernel(C **inhabited, uint *registred, uint *pos, uint size) {
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < size) {
        uint cid = inhabited[i]->cid;
        registred[cid] = 0;
        pos[cid] = 0;
    }
}

template<class A, class C>
void syncSpace(Society<A> *soc, CellularSpace<C> *cs, Neighborhood<A,C> *nb) {
    uint blocks;
    blocks = BLOCKS(soc->size);
    
    cudaMemset(nb->size, 0, sizeof(uint));
    findInhabitedKernel<<<blocks, THREADS>>>(soc->agents, soc->size, nb->inhabited, nb->registred, nb->size);
    uint nbSize;
    cudaMemcpy(&nbSize, nb->size, sizeof(uint), cudaMemcpyDeviceToHost);
    
    CHECK_ERROR
    if(nbSize > 0) {
    	
		scan(nb->inhabited, nb->offset, cs->quantities, nbSize);
		CHECK_ERROR

		blocks = BLOCKS(soc->size);
		sortAgentsKernel<<<blocks, THREADS>>>(soc->agents, soc->size, nb->neighborhood, nb->offset, nb->pos);
		CHECK_ERROR
		
		blocks = BLOCKS(nbSize);
		clearInfoKernel<<<blocks, THREADS>>>(nb->inhabited, nb->registred, nb->pos, nbSize);
		CHECK_ERROR
    }
}

#endif

