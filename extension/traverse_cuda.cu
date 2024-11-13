// clang-format off
#include "inc/common.h"
#include "inc/heapq.cuh"
#include "inc/bitmask.hpp"
// clang-format on

#define TILE_SIZE 4
#define TEAM_SIZE 8
#define N_WORKERS 4
#define WARP_SIZE 32
#define WARP_MASK 0xffffffff
#define BEAM_WIDTH 4
#define BUFFER_SIZE 256

// clang-format off
template <typename scalar_t, typename vector_t>
__device__ scalar_t _compute_dist_cuda(
    const scalar_t *storage, const scalar_t *query, index_t d_model,
    index_t location, index_t tile_idx, bool predicate
) {
    scalar_t distance = 0.0;
    for (auto offset = 0; predicate && offset < d_model; offset += WARP_SIZE) {
        auto local_idx = offset + tile_idx * TILE_SIZE;
        if (local_idx < d_model) {
            auto p = *reinterpret_cast<const vector_t *>(
                (const vector_t *)&storage[location * d_model + local_idx]
            );
            auto q = *reinterpret_cast<const vector_t *>(&query[local_idx]);
            distance += (q.x - p.x) * (q.x - p.x) + (q.y - p.y) * (q.y - p.y);
            distance += (q.z - p.z) * (q.z - p.z) + (q.w - p.w) * (q.w - p.w);
        }
    }
    __syncwarp();

    // reduce
    for (int offset = TEAM_SIZE / 2; offset > 0; offset /= 2) {
        distance += __shfl_down_sync(WARP_MASK, distance, offset, TEAM_SIZE);
    }

    return distance;
}

template <typename scalar_t, typename vector_t, int HEAP_SIZE>
__global__ void __launch_bounds__(BEAM_WIDTH *WARP_SIZE, 1)
    traverse_cuda_kernel(
        index_t *output_I, scalar_t *output_D, int8_t *bitmask,
        const index_t *indptr, const index_t *indices, const index_t *mapping,
        const scalar_t *storage, const scalar_t *query, const index_t *initial_I,
        const scalar_t *initial_D, int n_storage, int d_model, int n_neighbors,
        int n_initials, int ef_search, int32_t mask_value
    ) {
    // index
    auto block_idx = blockIdx.x;
    auto block_size = blockDim.y * WARP_SIZE;
    auto thread_idx = threadIdx.y * WARP_SIZE + threadIdx.x;

    // visited map
    auto visiting = [&](index_t node) -> bool {
        if (node < 0 || mapping[node] < 0) {
            return true;
        }
        auto n = mapping[node] / 4;
        auto r = mapping[node] % 4;
        auto offset = 4 * (n_storage >> 2);
        auto old = atomicOr(
            (int32_t *)&bitmask[
                block_idx * offset + 4 * n
            ],
            mask_value << 8 * r
        );
        return (0xff & (old >> 8 * r)) == mask_value;
    };

    // load query
    extern __shared__ scalar_t cached_query[];
    for (auto i = thread_idx; i < d_model; i += block_size) {
        cached_query[i] = query[block_idx * d_model + i];
    }

    // init
    __shared__ index_t buffer_I[BEAM_WIDTH][HEAP_SIZE];
    __shared__ scalar_t buffer_D[BEAM_WIDTH][HEAP_SIZE];
    __shared__ index_t openlist_I[BEAM_WIDTH][HEAP_SIZE];
    __shared__ scalar_t openlist_D[BEAM_WIDTH][HEAP_SIZE];
    for (auto i = threadIdx.x; i < HEAP_SIZE; i += WARP_SIZE) {
        buffer_I[threadIdx.y][i] = -1;
        buffer_D[threadIdx.y][i] = -INFINITY;
        openlist_I[threadIdx.y][i] = -1;
        openlist_D[threadIdx.y][i] = INFINITY;
    }
    for (auto i = threadIdx.y; i < n_initials; i += BEAM_WIDTH) {
        auto node = initial_I[block_idx * n_initials + i];
        auto distance = initial_D[block_idx * n_initials + i];
        if (node < 0 || distance >= -buffer_D[threadIdx.y][0]) {
            continue;
        }
        heap_replace<scalar_t, index_t>(
            openlist_D[threadIdx.y], openlist_I[threadIdx.y],
            HEAP_SIZE, distance, node
        );
        heap_pushpop<scalar_t, index_t>(
            buffer_D[threadIdx.y], buffer_I[threadIdx.y],
            HEAP_SIZE, -distance, node
        );
        visiting(node);
    }
    __syncthreads();

    // neighbors
    __shared__ index_t cached_neighbors[BUFFER_SIZE];
    auto fetch_neighbors = [&]() -> int {
        // pop
        index_t u = -1;
        while (true) {
            auto output = heap_pop<scalar_t, index_t>(
                openlist_D[threadIdx.y], openlist_I[threadIdx.y], HEAP_SIZE
            );
            if (u = output.value; u < 0) {
                break;
            }
            if (output.key >= -buffer_D[threadIdx.y][0]) {
                u = -1;
                break;
            }
            if (indptr[u] == indptr[u + 1]) {
                continue;
            }
            break;
        }

        // expand
        for (auto i = threadIdx.x; i < n_neighbors; i += WARP_SIZE) {
            auto offset = threadIdx.y * n_neighbors;
            if (u >= 0 && (indptr[u] + i < indptr[u + 1])) {
                if (auto v = indices[indptr[u] + i]; !visiting(v)) {
                    cached_neighbors[offset + i] = v;
                    continue;
                }
            }
            cached_neighbors[offset + i] = -1;
        }
        __syncthreads();

        // compact
        __shared__ int n_workloads;
        if (threadIdx.y == 0) {
            n_workloads = 0;

            // visit
            auto cursor = 0;
            for (auto i = threadIdx.x; i < BEAM_WIDTH * n_neighbors; i += WARP_SIZE) {
                auto v = cached_neighbors[i];
                auto ballot = __ballot_sync(WARP_MASK, v >= 0);
                if (ballot == 0) {
                    continue;
                }
                auto offset = __popc(
                    ((0x01 << threadIdx.x) - 1) & ballot
                );
                if (v >= 0) {
                    cached_neighbors[cursor + offset] = v;
                }
                cursor += __popc(ballot);
            }
            n_workloads = cursor;
        }
        __syncthreads();

        //
        return n_workloads;
    };

    // iterate
    for (auto step = 0; step < ef_search; step += 1) {
        // expand
        auto n_workloads = fetch_neighbors();

        // compute distances
        auto tile_idx = threadIdx.x % TEAM_SIZE;
        auto team_idx = threadIdx.x / TEAM_SIZE;
        auto worker_idx = threadIdx.y * N_WORKERS + team_idx;
        while (true) {
            // init
            auto predicate = worker_idx < n_workloads;
            if (__all_sync(WARP_MASK, predicate == false)) {
                break;
            }
            auto v = predicate ? cached_neighbors[worker_idx] : -1;

            // distance
            auto location = predicate ? mapping[v] : -1;
            predicate = (location < 0) ? false : predicate;
            auto distance = _compute_dist_cuda<scalar_t, vector_t>(
                storage, cached_query, d_model, location, tile_idx, predicate
            );

            // heapify
            for (auto i = 0; i < N_WORKERS; i += 1) {
                if (predicate && tile_idx == 0 && team_idx == i) {
                    heap_replace<scalar_t, index_t>(
                        openlist_D[threadIdx.y],
                        openlist_I[threadIdx.y],
                        HEAP_SIZE, distance, v
                    );
                    heap_pushpop<scalar_t, index_t>(
                        buffer_D[threadIdx.y],
                        buffer_I[threadIdx.y],
                        HEAP_SIZE, -distance, v
                    );
                }
                __syncwarp();
            }

            // advance
            worker_idx += BEAM_WIDTH * N_WORKERS;
        }
        __syncthreads();
    }
    __syncthreads();

    // store
    for (auto i = ef_search; i < HEAP_SIZE; i += 1) {
        heap_pop<scalar_t, index_t>(
            buffer_D[threadIdx.y], buffer_I[threadIdx.y], HEAP_SIZE
        );
    }
    for (auto i = 0; i < ef_search; i += 1) {
        auto output = heap_pop<scalar_t, index_t>(
            buffer_D[threadIdx.y], buffer_I[threadIdx.y], HEAP_SIZE
        );
        output_I[
            block_idx * BEAM_WIDTH * ef_search + threadIdx.y * ef_search + i
        ] = output.value;
        output_D[
            block_idx * BEAM_WIDTH * ef_search + threadIdx.y * ef_search + i
        ] = -output.key;
    }
}
// clang-format on

void traverse_cuda(
    torch::Tensor &output_I, torch::Tensor &output_D,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &mapping, const torch::Tensor &storage,
    const torch::Tensor &query, const torch::Tensor &initial_I,
    const torch::Tensor &initial_D, int n_neighbors
) {
    CHECK_CUDA(indptr, 1, torch::kInt64);
    CHECK_CUDA(indices, 1, torch::kInt64);
    CHECK_CUDA(mapping, 1, torch::kInt64);
    CHECK_CUDA(storage, 2, torch::kFloat32);
    CHECK_CUDA(query, 2, torch::kFloat32);
    CHECK_CUDA(initial_I, 2, torch::kInt64);
    CHECK_CUDA(initial_D, 2, torch::kFloat32);
    CHECK_CUDA(output_I, 3, torch::kInt64);
    CHECK_CUDA(output_D, 3, torch::kFloat32);

    // sizes
    auto d_model = query.size(-1);
    auto batch_size = query.size(0);
    auto n_nodes = indptr.size(0) - 1;
    auto n_storage = storage.size(0);
    auto ef_search = output_I.size(-1);
    auto n_initials = initial_I.size(-1);
    TORCH_CHECK(mapping.size(0) == n_nodes);
    TORCH_CHECK(storage.size(-1) == d_model);
    TORCH_CHECK(output_I.size(0) == batch_size);
    TORCH_CHECK(output_I.size(1) == BEAM_WIDTH);
    TORCH_CHECK(initial_I.size(0) == batch_size);
    TORCH_CHECK(output_I.sizes() == output_D.sizes());
    TORCH_CHECK(initial_I.sizes() == initial_D.sizes());
    TORCH_CHECK(BEAM_WIDTH * n_neighbors <= BUFFER_SIZE);
    TORCH_CHECK(n_neighbors % WARP_SIZE == 0);

    // bitmask
    auto n_elements = 4 * (n_storage >> 2);
    static std::shared_ptr<BitmaskPool<int8_t>> bitmask_pool;
    if (!bitmask_pool || bitmask_pool->batch_size() != batch_size ||
        bitmask_pool->n_elements() != n_elements) {
        std::cout << "[INFO] creating bitmask_pool" << std::endl;
        bitmask_pool = std::make_shared<BitmaskPool<int8_t>>(
            n_elements, batch_size, query.device()
        );
    }
    auto bitmask = bitmask_pool->get();

    // advance
    bitmask->advance();
    CHECK_CUDA(bitmask->tensor(), 2, torch::kInt8);
    TORCH_CHECK(bitmask->tensor().size(0) == batch_size);
    TORCH_CHECK(bitmask->tensor().size(-1) == n_elements);

    // dispatch
    auto stream = c10::cuda::getCurrentCUDAStream();
#define DISPATCH_KERNEL(HEAP_SIZE)                                          \
    do {                                                                    \
        auto smbytes = d_model * sizeof(float);                             \
        dim3 threads(WARP_SIZE, BEAM_WIDTH), blocks(batch_size);            \
        traverse_cuda_kernel<float, float4, HEAP_SIZE>                      \
            <<<blocks, threads, smbytes, stream.stream()>>>(                \
                output_I.data_ptr<index_t>(), output_D.data_ptr<float>(),   \
                bitmask->data_ptr(), indptr.data_ptr<index_t>(),            \
                indices.data_ptr<index_t>(), mapping.data_ptr<index_t>(),   \
                storage.data_ptr<float>(), query.data_ptr<float>(),         \
                initial_I.data_ptr<index_t>(), initial_D.data_ptr<float>(), \
                n_storage, d_model, n_neighbors, n_initials, ef_search,     \
                bitmask->marker()                                           \
            );                                                              \
        CUDA_CHECH(cudaGetLastError());                                     \
    } while (0)

    // launch
    if (ef_search <= 32) {
        DISPATCH_KERNEL(32);
    } else if (ef_search <= 64) {
        DISPATCH_KERNEL(64);
    } else if (ef_search <= 96) {
        DISPATCH_KERNEL(96);
    } else if (ef_search <= 128) {
        DISPATCH_KERNEL(128);
    } else if (ef_search <= 192) {
        DISPATCH_KERNEL(192);
    } else if (ef_search <= 256) {
        DISPATCH_KERNEL(256);
    } else {
        TORCH_CHECK("heapq_size not supported" && false);
    }
}