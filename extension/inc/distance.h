#ifndef DIST_HEADER_FILE_H
#define DIST_HEADER_FILE_H

// clang-format off
#include <immintrin.h>
#include "common.h"
// clang-format on

#define SIMD_WIDTH 8

inline void prefetch_l1(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T0);
}

inline void prefetch_l2(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T1);
}

inline void prefetch_l3(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_T2);
}

inline void prefetch_rw(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_ET0);
}

inline void prefetch_nta(const void* address) {
    _mm_prefetch((const char*)address, _MM_HINT_NTA);
}

inline float compute_dist_avx2(
    const float* query, const float* storage, index_t v, index_t d_model,
    index_t offset = 0
) {
    TORCH_CHECK(offset % SIMD_WIDTH == 0);
    TORCH_CHECK(d_model % SIMD_WIDTH == 0);

    // window
    auto buffer = _mm256_setzero_ps();
    while (offset < d_model) {
        // query
        auto query_reg = _mm256_load_ps(query + offset);

        // compute
        auto tmp_reg = _mm256_sub_ps(
            query_reg, _mm256_load_ps(storage + v * d_model + offset)
        );
        buffer = _mm256_fmadd_ps(tmp_reg, tmp_reg, buffer);
        offset += SIMD_WIDTH;
    }

    // reduce
    auto reduced = 0.0f;
    for (auto k = 0; k < SIMD_WIDTH; k += 1) {
        reduced += buffer[k];
    }
    return reduced;
}

// clang-format off
inline void compute_dist_avx2r(
    float* distances, const float* query, const float* storage,
    const index_t* indices, index_t d_model, index_t n_elements
) {
    for (auto i = 0; i < n_elements; i += 1) {
        distances[i] = compute_dist_avx2(
            query, storage, indices[i], d_model
        );
    }
}
// clang-format on

// clang-format off
template <int TILE_SIZE>
inline void compute_dist_avx2c(
    float* distances, const float* query, const float* storage,
    const index_t* indices, index_t d_model
) {
    // init
    __m256 buffer[TILE_SIZE];
    for (auto i = 0; i < TILE_SIZE; i += 1) {
        buffer[i] = _mm256_setzero_ps();
    }

    // window
    for (auto offset = 0; offset < d_model; offset += SIMD_WIDTH) {
        // query
        auto query_reg = _mm256_load_ps(query + offset);

        // compute
#pragma unroll
        for (auto i = 0; i < TILE_SIZE; i += 1) {
            auto tmp_reg = _mm256_sub_ps(
                query_reg, _mm256_load_ps(
                    storage + indices[i] * d_model + offset
                )
            );
            buffer[i] = _mm256_fmadd_ps(tmp_reg, tmp_reg, buffer[i]);
        }
    }

    // reduce
#pragma unroll
    for (auto i = 0; i < TILE_SIZE; i += 1) {
        auto reduced = 0.0f;
        for (auto k = 0; k < SIMD_WIDTH; k += 1) {
            reduced += buffer[i][k];
        }
        distances[i] = reduced;
    }
}
// clang-format on

#endif
