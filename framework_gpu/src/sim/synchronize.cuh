#ifndef SYNCHRONIZE_CUH
#define SYNCHRONIZE_CUH

template<class A, class C>
void synchronize(Society<A> *soc, CellularSpace<C> *cs, Neighborhood<A,C> *nb) {	
    syncAgents(soc, cs);
    syncSpace(soc, cs, nb);
}

#endif
