// clang-format off
#include "inc/common.h"
// clang-format on

#define BEAM_WIDTH 4
#define N_PREFETCH 2

using Slice = torch::indexing::Slice;
using CUDAStream = c10::cuda::CUDAStream;

// clang-format off
void traverse_refine(
    torch::Tensor &output_I, torch::Tensor &output_D,
    torch::Tensor &buffer_I, torch::Tensor &buffer_D,
    torch::Tensor &initial_I, torch::Tensor &initial_D,
    const std::vector<torch::Tensor> &subgraph,
    const std::vector<torch::Tensor> &fullgraph,
    const torch::Tensor &storage, const torch::Tensor &query,
    int ef_search, int d_principle
);

void traverse_cuda(
    torch::Tensor &output_I, torch::Tensor &output_D,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &mapping, const torch::Tensor &storage,
    const torch::Tensor &query, const torch::Tensor &initial_I,
    const torch::Tensor &initial_D, int n_neighbors
);
// clang-format on

// clang-format off
std::vector<torch::Tensor> pipeline(
    const std::vector<torch::Tensor> &graph,
    const std::vector<torch::Tensor> &subgraph,
    const std::vector<torch::Tensor> &subgraph_cuda,
    const torch::Tensor &storage, const torch::Tensor &storage_cuda,
    const torch::Tensor &query, const torch::Tensor &query_cuda,
    const torch::Tensor &initial_I, const torch::Tensor &initial_D,
    int n_neighbors, int ef_search, int d_principle, int k, int chunk_size
) {
    TORCH_CHECK(graph.size() == 2);
    TORCH_CHECK(subgraph.size() == 2);
    TORCH_CHECK(subgraph_cuda.size() == 3);
    CHECK_CPU(query, 2, torch::kFloat32);
    CHECK_CPU(storage, 2, torch::kFloat32);
    CHECK_CPU(graph[0], 1, torch::kInt64);
    CHECK_CPU(graph[1], 1, torch::kInt64);
    CHECK_CPU(subgraph[0], 1, torch::kInt64);
    CHECK_CPU(subgraph[1], 1, torch::kInt64);
    CHECK_CUDA(query_cuda, 2, torch::kFloat32);
    CHECK_CUDA(storage_cuda, 2, torch::kFloat32);
    CHECK_CUDA(subgraph_cuda[0], 1, torch::kInt64);
    CHECK_CUDA(subgraph_cuda[1], 1, torch::kInt64);
    CHECK_CUDA(subgraph_cuda[2], 1, torch::kInt64);

    // sizes
    auto batch_size = query.size(0);
    TORCH_CHECK(batch_size % chunk_size == 0);
    auto topk_I = torch::empty(
        {batch_size, k}, torch::dtype(torch::kInt64)
    );
    auto topk_D = torch::empty(
        {batch_size, k}, torch::dtype(torch::kFloat32)
    );

    // streams
    static std::vector<CUDAStream> streams;
    if (streams.empty()) {
        std::cout << "[INFO] init cuda streams" << std::endl;
        for (auto i = 0; i < N_PREFETCH; i += 1) {
            streams.emplace_back(c10::cuda::getStreamFromPool());
        }
    }
    auto default_stream = c10::cuda::getCurrentCUDAStream();
    default_stream.synchronize();

    // buffers
    auto dev_buffer_I = std::vector<torch::Tensor>();
    auto dev_buffer_D = std::vector<torch::Tensor>();
    auto host_buffer_I = std::vector<torch::Tensor>();
    auto host_buffer_D = std::vector<torch::Tensor>();
    auto pinned_option = query.options().pinned_memory(true);
    for (auto i = 0; i < N_PREFETCH; i += 1) {
        dev_buffer_I.push_back(torch::empty(
            {chunk_size, BEAM_WIDTH, ef_search}, initial_I.options()
        ));
        dev_buffer_D.push_back(torch::empty(
            {chunk_size, BEAM_WIDTH, ef_search}, initial_D.options()
        ));
        host_buffer_I.push_back(torch::empty(
            {chunk_size, BEAM_WIDTH * ef_search}, pinned_option.dtype(torch::kInt64)
        ));
        host_buffer_D.push_back(torch::empty(
            {chunk_size, BEAM_WIDTH * ef_search}, pinned_option.dtype(torch::kFloat32)
        ));
    }

    // pipeline
    auto n_chunks = batch_size / chunk_size;
    for (auto i = 0; i < n_chunks + N_PREFETCH; i += 1) {
        auto slot = i % N_PREFETCH;

        // process
        if (i >= N_PREFETCH) {
            streams[slot].synchronize();
            auto left = (i - N_PREFETCH) * chunk_size;
            auto right = left + chunk_size;
            auto I = topk_I.index({Slice(left, right)});
            auto D = topk_D.index({Slice(left, right)});
            traverse_refine(
                I, D,
                host_buffer_I[slot], host_buffer_D[slot],
                host_buffer_I[slot], host_buffer_D[slot],
                subgraph, graph, storage,
                query.index({Slice(left, right)}),
                ef_search, d_principle
            );
        }

        // prefetch
        if (i < n_chunks) {
            auto left = i * chunk_size;
            auto right = left + chunk_size;
            auto _ = c10::cuda::CUDAStreamGuard(streams[slot]);
            traverse_cuda(
                dev_buffer_I[slot], dev_buffer_D[slot],
                subgraph_cuda[0], subgraph_cuda[1],
                subgraph_cuda[2], storage_cuda,
                query_cuda.index({Slice(left, right)}),
                initial_I.index({Slice(left, right)}),
                initial_D.index({Slice(left, right)}),
                n_neighbors
            );
            host_buffer_I[slot].copy_(
                dev_buffer_I[slot].view({chunk_size, -1}), true
            );
            host_buffer_D[slot].copy_(
                dev_buffer_D[slot].view({chunk_size, -1}), true
            );
        }
    }

    //
    return {topk_I, topk_D};
}
// clang-format on
