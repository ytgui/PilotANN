#ifndef HEAPQ_HEADER_FILE_H
#define HEAPQ_HEADER_FILE_H

#include "common.h"

template <typename T>
__host__ __device__ void inline __swap(T *a, T *b) {
    T c(*a);
    *a = *b;
    *b = c;
}

template <typename K, typename V>
__host__ __device__ void __heapify_up(
    K *keys, V *values, int hsize
) {
    for (auto cursor = 0, child = 1; child < hsize;
         cursor = child, child = 2 * cursor + 1) {
        auto right = child + 1;
        if (right < hsize && keys[right] < keys[child]) {
            child = right;
        };
        if (keys[cursor] <= keys[child]) {
            break;
        }
        __swap(&keys[cursor], &keys[child]);
        __swap(&values[cursor], &values[child]);
    }
}

template <typename K, typename V>
struct pair_t {
    K key;
    V value;
};

template <typename K, typename V>
__host__ __device__ pair_t<K, V> heap_pop(
    K *keys, V *values, int hsize, K inf = INFINITY
) {
    auto output = pair_t<K, V>{.key = keys[0], .value = values[0]};
    keys[0] = inf, values[0] = -1;
    __heapify_up<K, V>(keys, values, hsize);
    return output;
}

template <typename K, typename V>
__host__ __device__ void heap_replace(
    K *keys, V *values, int hsize, K k, V v
) {
    // find a slot
    auto cursor = 0;
    for (auto child = 1; child < hsize;
         cursor = child, child = 2 * cursor + 1) {
        auto right = child + 1;
        if (right < hsize && keys[right] > keys[child]) {
            child = right;
        };
    }

    // new item is more larger
    if (k >= keys[cursor]) {
        return;
    }
    keys[cursor] = k, values[cursor] = v;

    // heapify down
    while (cursor > 0) {
        auto parent = (cursor - 1) / 2;
        if (keys[parent] <= keys[cursor]) {
            break;
        }
        __swap(&keys[cursor], &keys[parent]);
        __swap(&values[cursor], &values[parent]);
        cursor = parent;
    }
}

template <typename K, typename V>
__host__ __device__ void heap_pushpop(
    K *keys, V *values, int hsize, K k, V v
) {
    if (k <= keys[0]) {
        return;
    }
    keys[0] = k, values[0] = v;
    __heapify_up<K, V>(keys, values, hsize);
}

#endif