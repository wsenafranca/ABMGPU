#ifndef ITERATOR_CUH
#define ITERATOR_CUH

template<class T>
class Iterator{
public:
    __device__ Iterator() {}
    
    __device__ Iterator(const Iterator &it) : p(it.p) {}
    
    __device__ Iterator(T *ptr) : p(ptr) {}
    
    __device__ virtual Iterator& operator++() {p++;return *this;}
    
    __device__ virtual Iterator operator++(int) {Iterator tmp(*this); operator++(); return tmp;}
    
    __device__ virtual Iterator operator+(int i) {return Iterator(p+i);}
    
    //__device__ virtual const Iterator& operator=(const Iterator &it) {p = it.p; return *this;}
    
    __device__ virtual bool operator==(const Iterator& rhs) {return p==rhs.p;}
    
    __device__ virtual bool operator!=(const Iterator& rhs) {return p!=rhs.p;}
    
    __device__ virtual T& operator*() {return *p;}
    
    T *p;
};

#endif
