// clang-format off
#include <vector>
#include "inc/common.h"
#include "inc/heapq.cuh"
// clang-format on

#define BLOCK_SIZE 32

// clang-format off
template <typename scalar_t>
__global__ void routing_cuda_kernel(
    scalar_t *distances, const int64_t *router,
    const scalar_t *entry_vectors, const scalar_t *query,
    int batch_size, int d_model, int n_leaves
) {
    // index
    auto block_idx = blockIdx.x;

    // n window
    for (auto offset_n = 0; offset_n < n_leaves; offset_n += BLOCK_SIZE) {
        // k window
        __shared__ scalar_t cache_rhs[BLOCK_SIZE][BLOCK_SIZE];
        for (auto offset_k = 0; offset_k < d_model; offset_k += BLOCK_SIZE) {
            // load rhs
            auto leave_idx = offset_n + threadIdx.y;
            auto global_idx = block_idx * n_leaves * d_model;
            if (offset_k + threadIdx.x < d_model) {
                cache_rhs[threadIdx.y][threadIdx.x] = entry_vectors[
                    global_idx + leave_idx * d_model + (offset_k + threadIdx.x)
                ];
            } else {
                cache_rhs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();

            // m window
            __shared__ scalar_t cache_lhs[BLOCK_SIZE][BLOCK_SIZE];
            for (auto i = threadIdx.y; i < batch_size; i += BLOCK_SIZE) {
                if (router[i] != block_idx) {
                    continue;
                }

                // load lhs
                if (offset_k + threadIdx.x < d_model) {
                    cache_lhs[threadIdx.y][threadIdx.x] = query[
                        i * d_model + offset_k + threadIdx.x
                    ];
                } else {
                    cache_lhs[threadIdx.y][threadIdx.x] = 0.0f;
                }
                __syncwarp();

                // reduce
                auto reduced = 0.0f;
                for (auto k = 0; k < BLOCK_SIZE; k += 1) {
                    auto q = cache_lhs[threadIdx.y][k];
                    auto p = cache_rhs[threadIdx.x][k];
                    reduced += (q - p) * (q - p);
                }
                auto leave_idx = offset_n + threadIdx.x;
                distances[i * n_leaves + leave_idx] += reduced;
            }
            __syncthreads();
        }
    }
}
// clang-format on

// clang-format off
std::vector<torch::Tensor> routing_cuda(
    const torch::Tensor &query,
    const torch::Tensor &route_vectors,
    const torch::Tensor &entry_nodes,
    const torch::Tensor &entry_vectors
) {
    CHECK_CUDA(query, 2, torch::kFloat32);
    CHECK_CUDA(route_vectors, 2, torch::kFloat32);
    CHECK_CUDA(entry_vectors, 3, torch::kFloat32);
    CHECK_CUDA(entry_nodes, 2, torch::kInt64);

    // sizes
    auto d_model = query.size(-1);
    auto batch_size = query.size(0);
    auto n_leaves = entry_nodes.size(-1);
    auto n_routers = route_vectors.size(0);
    TORCH_CHECK(route_vectors.size(-1) == d_model);
    TORCH_CHECK(entry_vectors.size(-1) == d_model);
    TORCH_CHECK(entry_vectors.size(0) == n_routers);
    TORCH_CHECK(entry_vectors.size(1) == n_leaves);
    TORCH_CHECK(n_leaves % BLOCK_SIZE == 0);

    // output
    auto router = torch::argmin(
        torch::cdist(query, route_vectors), -1, true
    );
    auto distances = torch::zeros(
        {batch_size, n_leaves}, query.options()
    );

    // launch
    dim3 blocks(n_routers);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    auto stream = c10::cuda::getCurrentCUDAStream();
    routing_cuda_kernel<float><<<blocks, threads, 0, stream.stream()>>>(
        distances.data_ptr<float>(), router.data_ptr<int64_t>(),
        entry_vectors.data_ptr<float>(), query.data_ptr<float>(),
        batch_size, d_model, n_leaves
    );
    CUDA_CHECH(cudaGetLastError());

    //
    return {router.flatten(), distances};
}
// clang-format on
