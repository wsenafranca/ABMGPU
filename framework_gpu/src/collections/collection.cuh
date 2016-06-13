#ifndef COLLECTION_H
#define COLLECTION_H

class Collection{
public:
    Collection(uint Size, const uint Capacity) : memPtr(0), size(Size), capacity(Capacity), bytes(0){}
    virtual ~Collection() {}
    
    virtual Event* initializeEvent() = 0;
    
    uint getCapacity() const {return capacity;}
    
    uint getSize() const {return size;}
    void setSize(uint size) {this->size = size;}
    
    void setMemPtrDevice(unsigned char *ptr) {memPtr = ptr;}
    unsigned char* getMemPtrDevice() const {return memPtr;}
    
    uint getIndexOnMemoryDevice() const {return index;}
    void setIndexOnMemoryDevice(uint index) {this->index = index;}
    
    uint alloc(uint bytes) {uint b = this->bytes; this->bytes += bytes; return b;}
    unsigned long getBytes() {return bytes;}
    
//private:
    unsigned char *memPtr;
    const uint capacity;
    uint size;
    uint index;
    unsigned long bytes;
};

#endif

