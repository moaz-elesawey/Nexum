#ifndef _NxMEMORY_H_
#define _NxMEMORY_H_

#include "NxCore.h"

/// Maximum size of memory block to allocate.
#define NxARENA_MAX_SIZE 1024*1024*256

typedef struct NxRegion {
    struct NxRegion* next; ///< pointer to the next memory block.
    u64 count; ///< number of items allocated in the block.
    u64 capacity; ///< the maximum size of the memory block.
    u64 size; ///< the current size occupied in the block.
    NxDTYPE* data; ///< the data of the memory block.
} NxRegion;

typedef struct NxArena {
    NxRegion* begin; ///< pointer to the first memory block int the Arena.
    NxRegion* end; ///< pointer to the last memory block in the Arena.
} NxArena;

#endif /* _NxMEMORY_H_ */