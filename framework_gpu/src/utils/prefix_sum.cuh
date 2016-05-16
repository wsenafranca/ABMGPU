#ifndef PREFIXSUM_CUH
#define PREFIXSUM_CUH

uint next2k(uint n) {
    n--; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; n++;
    return n;
}

__global__ void midscan(uint *in, uint *out, uint *buffer, uint n) {
    extern __shared__ uint s_data[];
    uint tid = threadIdx.x;
    uint i = threadIdx.x+blockDim.x*blockIdx.x;
    uint offset = 1;
    s_data[tid] = (i < n) ? in[i] : 0;
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
    if(i < n)
    	out[i] = s_data[tid];
}

__global__ void postscan(uint *buffer, uint *offset, uint len) {
	__shared__ uint buf;
    uint i = threadIdx.x + blockDim.x*blockIdx.x;
    if(i < len) 
        buf = buffer[blockIdx.x];
        
    __syncthreads();
    
    if(i < len) 
        offset[i]+=buf;
}

void recursiveScan(uint *d_in, uint *d_out, uint len) {
	uint size = next2k(len);
	uint blocks = BLOCKS(size);
	uint *d_buffer;
	cudaMalloc(&d_buffer, sizeof(uint)*blocks);
	uint threads = min(THREADS, size);
	midscan<<<blocks, threads, sizeof(uint)*(size/blocks)>>>(d_in, d_out, d_buffer, len);
	if(blocks > THREADS)
		recursiveScan(d_buffer, d_buffer, blocks);
	else
		midscan<<<1, blocks, sizeof(uint)*blocks>>>(d_buffer, d_buffer, NULL, blocks);
	postscan<<<blocks, threads, sizeof(uint)>>>(d_buffer, d_out, len);
	cudaFree(d_buffer);
}

#endif

