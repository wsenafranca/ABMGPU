#ifndef EXECUTE_CUH
#define EXECUTE_CUH

typedef void(*AgentExecute)(Agent*);
typedef void(*AgentNeighborhoodExecute)(Agent*, Agent*, const NeighborhoodIterator&, const NeighborhoodIterator&);
typedef void(*AgentSpatiallyExecute)(Agent*, Cell*, uint, uint);
typedef void(*AgentSpatiallyNeighborhoodExecute)(Agent*, Agent*, Cell*, uint, uint, const NeighborhoodIterator&, const NeighborhoodIterator&);
typedef void(*CellChange)(Cell*);
typedef void(*CellSpatiallyChange)(Cell*, Cell*, uint, uint);

#endif
