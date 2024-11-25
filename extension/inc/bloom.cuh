#ifndef BLOOM_HEADER_FILE_H
#define BLOOM_HEADER_FILE_H

#include "common.h"

#define FNV_PRIME 0x01000193
#define FNV_OFFSET 0x811c9dc5

__device__ uint32_t fnv1a_32(uint32_t x, uint32_t offset) {
    uint32_t h = offset;
    for (auto i = 0; i < 4; i += 1) {
        h ^= x >> (8 * i);
        h *= FNV_PRIME;
    }
    return h;
}

#endif
