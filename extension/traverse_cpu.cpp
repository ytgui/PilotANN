// clang-format off
#include "inc/common.h"
#include "inc/heapq.cuh"
#include "inc/bitmask.hpp"
#include "inc/distance.h"
// clang-format on

#define TILE_SIZE 4

// clang-format off
template <int HEAP_SIZE>
void traverse_cpu_kernel(
    index_t *output_I, float *output_D, int32_t *bitmask, const index_t *indptr,
    const index_t *indices, const float *storage, const float *query,
    const index_t *initial_I, const float *initial_D, int n_storage,
    int d_model, int n_initials, int ef_search, int k, int32_t mask_value
) {
    // heap
    auto buffer_D = std::array<float, HEAP_SIZE>();
    auto buffer_I = std::array<index_t, HEAP_SIZE>();
    auto openlist_D = std::array<float, 2 * HEAP_SIZE>();
    auto openlist_I = std::array<index_t, 2 * HEAP_SIZE>();
    auto heap_update = [&](const float *Ds, const index_t *Vs, int n) {
        for (auto i = 0; i < n; i += 1) {
            if (Ds[i] >= -buffer_D[0]) {
                continue;
            }
            heap_pushpop<float, index_t>(
                buffer_D.data(), buffer_I.data(), HEAP_SIZE, -Ds[i], Vs[i]
            );
            heap_replace<float, index_t>(
                openlist_D.data(), openlist_I.data(), 2 * HEAP_SIZE, Ds[i],
                Vs[i]
            );
        }
    };

    // init
    buffer_I.fill(-1);
    openlist_I.fill(-1);
    buffer_D.fill(-INFINITY);
    openlist_D.fill(INFINITY);
    for (auto i = 0; i < n_initials; i += 1) {
        if (auto u = initial_I[i]; u >= 0) {
            prefetch_l2(&bitmask[u]);
        }
    }
    for (auto i = 0; i < n_initials; i += 1) {
        auto u = initial_I[i];
        if (u < 0 || u >= n_storage) {
            break;
        }
        if (bitmask[u] == mask_value) {
            continue;
        }
        if (bitmask[u] != -1) {
            bitmask[u] = mask_value;
        }
        auto d = initial_D[i];
        heap_replace<float, index_t>(
            openlist_D.data(), openlist_I.data(), 2 * HEAP_SIZE, d, u
        );
    }

    // iterate
    for (auto step = 0; step < 2 * ef_search; step += 1) {
        // seed
        auto u = heap_pop<float, index_t>(
            openlist_D.data(), openlist_I.data(), 2 * HEAP_SIZE
        );
        if (u.value < 0 || u.key >= -buffer_D[0]) {
            break;
        }

        // prefetch
        auto left = indptr[u.value];
        auto right = indptr[u.value + 1];
        for (auto i = left; i < right; i += 1) {
            if (auto v = indices[i]; v >= 0) {
                prefetch_l2(&bitmask[v]);
            }
        }

        // expand
        int cursor = 0;
        float Ds[TILE_SIZE];
        index_t Vs[TILE_SIZE];
        for (auto i = left; i < right; i += 1) {
            auto v = indices[i];
            if (v < 0) {
                break;
            }
            auto visited = bitmask[v];
            if (visited == -1 || visited == mask_value) {
                continue;
            }
            bitmask[v] = mask_value;
            Vs[cursor] = v;
            cursor += 1;

            // compute
            prefetch_l2(storage + v * d_model);
            if (cursor == TILE_SIZE) {
                compute_dist_avx2c<TILE_SIZE>(
                    Ds, query, storage, Vs, d_model
                );
                heap_update(Ds, Vs, cursor);
                cursor = 0;
            }
        }

        // remains
        if (cursor > 0) {
            compute_dist_avx2r(
                Ds, query, storage, Vs, d_model, cursor
            );
            heap_update(Ds, Vs, cursor);
        }
    }

    // merge
    for (auto i = 0; i < n_initials; i += 1) {
        auto u = initial_I[i];
        if (u < 0 || u >= n_storage) {
            break;
        }
        auto d = initial_D[i];
        heap_pushpop<float, index_t>(
            buffer_D.data(), buffer_I.data(), HEAP_SIZE, -d, u
        );
    }

    // top-k
    for (auto i = k; i < HEAP_SIZE; i += 1) {
        heap_pop<float, index_t>(
            buffer_D.data(), buffer_I.data(), HEAP_SIZE
        );
    }
    for (auto i = 0; i < k; i += 1) {
        auto output = heap_pop<float, index_t>(
            buffer_D.data(), buffer_I.data(), HEAP_SIZE
        );
        output_I[k - i - 1] = output.value;
        output_D[k - i - 1] = -output.key;
    }
}
// clang-format on

void traverse_cpu(
    torch::Tensor &output_I, torch::Tensor &output_D,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &storage, const torch::Tensor &query,
    const torch::Tensor &initial_I, const torch::Tensor &initial_D,
    int n_neighbors, int ef_search
) {
    CHECK_CPU(query, 2, torch::kFloat32);
    CHECK_CPU(indptr, 1, torch::kInt64);
    CHECK_CPU(indices, 1, torch::kInt64);
    CHECK_CPU(storage, 2, torch::kFloat32);
    CHECK_CPU(initial_I, 2, torch::kInt64);
    CHECK_CPU(initial_D, 2, torch::kFloat32);
    CHECK_CPU(output_I, 2, torch::kInt64);
    CHECK_CPU(output_D, 2, torch::kFloat32);

    // size
    auto k = output_I.size(-1);
    auto d_model = query.size(-1);
    auto batch_size = query.size(0);
    auto n_storage = storage.size(0);
    auto n_initials = initial_I.size(-1);
    TORCH_CHECK(storage.size(-1) == d_model);
    TORCH_CHECK(output_I.size(0) == batch_size);
    TORCH_CHECK(initial_I.size(0) == batch_size);
    TORCH_CHECK(output_I.sizes() == output_D.sizes());
    TORCH_CHECK(initial_I.sizes() == initial_D.sizes());

    // bitmask
    static std::shared_ptr<BitmaskPool<int32_t>> bitmask_pool;
    if (!bitmask_pool || bitmask_pool->n_elements() != n_storage) {
        std::cout << "[INFO] creating bitmask_pool" << std::endl;
        bitmask_pool = std::make_shared<BitmaskPool<int32_t>>(n_storage, 1);
    }

    // process
#pragma omp parallel for schedule(dynamic)
    for (auto i = 0; i < batch_size; i += 1) {
        // init
        auto bitmask = bitmask_pool->get();

        // advance
        bitmask->advance();
        CHECK_CPU(bitmask->tensor(), 2, torch::kInt32);
        TORCH_CHECK(bitmask->tensor().size(-1) == n_storage);

        // dispatch
#define DISPATCH_KERNEL(HEAP_SIZE, I)                                         \
    do {                                                                      \
        traverse_cpu_kernel<HEAP_SIZE>(                                       \
            output_I.data_ptr<index_t>() + I * k,                             \
            output_D.data_ptr<float>() + I * k, bitmask->data_ptr(),          \
            indptr.data_ptr<index_t>(), indices.data_ptr<index_t>(),          \
            storage.data_ptr<float>(), query.data_ptr<float>() + I * d_model, \
            initial_I.data_ptr<index_t>() + I * n_initials,                   \
            initial_D.data_ptr<float>() + I * n_initials, n_storage, d_model, \
            n_initials, ef_search, k, bitmask->marker()                       \
        );                                                                    \
    } while (0)

        // launch
        if (ef_search <= 16) {
            DISPATCH_KERNEL(16, i);
        } else if (ef_search <= 32) {
            DISPATCH_KERNEL(32, i);
        } else if (ef_search <= 64) {
            DISPATCH_KERNEL(64, i);
        } else if (ef_search <= 128) {
            DISPATCH_KERNEL(128, i);
        } else if (ef_search <= 192) {
            DISPATCH_KERNEL(192, i);
        } else if (ef_search <= 256) {
            DISPATCH_KERNEL(256, i);
        } else {
            TORCH_CHECK("ef_search not supported" && false);
        }
        bitmask_pool->put(bitmask);
    }
}
