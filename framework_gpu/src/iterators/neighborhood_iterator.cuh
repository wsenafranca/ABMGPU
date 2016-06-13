#ifndef NEIGHBORHOOD_ITERATOR_CUH
#define NEIGHBORHOOD_ITERATOR_CUH

class NeighborhoodIterator {
public:
    __device__ NeighborhoodIterator() {}
    
    __device__ NeighborhoodIterator(uint *ptr) : p(ptr) {}
    
    __device__ NeighborhoodIterator(const NeighborhoodIterator &it) : p(it.p), iBegin(it.iBegin), iEnd(it.iEnd){}
    
    __device__ NeighborhoodIterator(const Iterator<uint> &it) : p(it.p) {}
    
    __device__ NeighborhoodIterator(const Iterator<uint> &it, Iterator<uint> *begin, Iterator<uint> *end) 
                    : p(it), iBegin(begin), iEnd(end) {}
    
    __device__ NeighborhoodIterator& operator++() {
        ++p;
        if(p == *iEnd) {
            ++iEnd;
            ++iBegin;
            p = *iBegin;
        }
        return *this;
    }
    
    __device__ virtual NeighborhoodIterator operator++(int) {NeighborhoodIterator tmp(*this); operator++(); return tmp;}
    
    __device__ virtual bool operator==(const NeighborhoodIterator& rhs) {return iBegin==rhs.iBegin;}
    
    __device__ virtual bool operator!=(const NeighborhoodIterator& rhs) {return iBegin!=rhs.iBegin;}
    
    __device__ virtual uint& operator*() {return *p;}
    
    Iterator<uint> p;
    Iterator< Iterator<uint> > iBegin;
    Iterator< Iterator<uint> > iEnd;
    
    __device__ __forceinline__ static void create(Agent *ag, uint *neighborhood, uint n, uint m, uint nxdim, uint nydim, 
                                  Iterator<uint> *begins, Iterator<uint> *ends, uint *neighs) 
    {
        uint2 pos = ag->cell->getPos();
        int x = truncf(pos.x/m);
        int y = truncf(pos.y/n);
        *neighs = 0;
        for(int ny = y-1; ny <= y+1; ny++) {
            for(int nx = x-1; nx <= x+1; nx++) {
                if(nx >= 0 && nx < nxdim && ny >= 0 && ny < nydim) {
                    uint begin = tex2D(beginsRef, nx, ny);
                    uint end = tex2D(endsRef, nx, ny);
                    if(end > 0) {
                        begins[*neighs] = Iterator<uint>(&neighborhood[begin]);
                        ends[*neighs] = Iterator<uint>(&neighborhood[end]);
                        (*neighs)++;
                    }
                }
            }
        }
    }
};

#endif

