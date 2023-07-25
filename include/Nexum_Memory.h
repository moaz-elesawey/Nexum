#ifndef _Nexum_MEMORY_H_
#define _Nexum_MEMORY_H_

#include "Nexum_Core.h"

/// Maximum size of memory block to allocate.
#define Nexum_ARENA_MAX_SIZE 1024*1024*256

typedef struct Nexum_Region {
    struct Nexum_Region* next; ///< pointer to the next memory block.
    u64 count; ///< number of items allocated in the block.
    u64 capacity; ///< the maximum size of the memory block.
    u64 size; ///< the current size occupied in the block.
    Nexum_DTYPE* data; ///< the data of the memory block.
} Nexum_Region;

typedef struct Nexum_Arena {
    Nexum_Region* begin; ///< pointer to the first memory block int the Arena.
    Nexum_Region* end; ///< pointer to the last memory block in the Arena.
} Nexum_Arena;

#endif /* _Nexum_MEMORY_H_ */