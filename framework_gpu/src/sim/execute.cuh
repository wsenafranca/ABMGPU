#ifndef EXECUTE_CUH
#define EXECUTE_CUH

typedef void(*SocialChange)(Agent*);
typedef void(*SocialChangePair)(Agent*, Agent*);
typedef void(*SpaceSocialChange)(Agent*, Cell*, uint, uint);
typedef void(*SpaceSocialChangePair)(Agent*, Agent*, Cell*, uint, uint);
typedef void(*SpatialChange)(Cell*);
typedef void(*SpatialChange)(Cell*);

#endif
