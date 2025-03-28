// clang-format off
#include "inc/common.h"
// clang-format on

/***** basic operators *****/
void heapq_pop_cpu(torch::Tensor &keys, torch::Tensor &values);
void heapq_pushpop_cpu(
    torch::Tensor &keys, torch::Tensor &values, float k, index_t v
);
void heapq_replace_cpu(
    torch::Tensor &keys, torch::Tensor &values, float k, index_t v
);
torch::Tensor bitmask_put_cpu(const torch::Tensor &keys, index_t n);
torch::Tensor square_dist_cpu(
    const torch::Tensor &query, const torch::Tensor &storage,
    const torch::Tensor &nodelist
);

/***** sampling operators *****/
std::vector<std::vector<index_t>> sampling_cpu(
    const std::vector<index_t> &indptr, const std::vector<index_t> &indices,
    size_t n_samples, int n_hops
);

/***** traversal operators *****/
void traverse_cpu(
    torch::Tensor &output_I, torch::Tensor &output_D, torch::Tensor &initial_I,
    torch::Tensor &initial_D, const torch::Tensor &indptr,
    const torch::Tensor &indices, const torch::Tensor &storage,
    const torch::Tensor &query, int ef_search
);
void traverse_cuda(
    torch::Tensor &output_I, torch::Tensor &output_D,
    const torch::Tensor &indptr, const torch::Tensor &indices,
    const torch::Tensor &mapping, const torch::Tensor &storage,
    const torch::Tensor &query, const torch::Tensor &initial_I,
    const torch::Tensor &initial_D, int n_neighbors
);
void traverse_refine(
    torch::Tensor &output_I, torch::Tensor &output_D, torch::Tensor &buffer_I,
    torch::Tensor &buffer_D, torch::Tensor &initial_I, torch::Tensor &initial_D,
    const std::vector<torch::Tensor> &subgraph,
    const std::vector<torch::Tensor> &fullgraph, const torch::Tensor &storage,
    const torch::Tensor &query, int ef_search, int d_principle
);

/***** pipeline operator *****/
std::vector<torch::Tensor> pipeline(
    const std::vector<torch::Tensor> &graph,
    const std::vector<torch::Tensor> &subgraph,
    const std::vector<torch::Tensor> &subgraph_cuda,
    const torch::Tensor &storage, const torch::Tensor &storage_cuda,
    const torch::Tensor &query, const torch::Tensor &query_cuda,
    const torch::Tensor &initial_I, const torch::Tensor &initial_D,
    int n_neighbors, int ef_search, int d_principle, int k, int chunk_size
);

/***** routing operator *****/
std::vector<torch::Tensor> routing_cuda(
    const torch::Tensor &query, const torch::Tensor &route_vectors,
    const torch::Tensor &entry_nodes, const torch::Tensor &entry_vectors
);

/***** pybind11 module *****/
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // basic
    m.def("heapq_pop_cpu", &heapq_pop_cpu, "heapq_pop_cpu");
    m.def("heapq_pushpop_cpu", &heapq_pushpop_cpu, "heapq_pushpop_cpu");
    m.def("heapq_replace_cpu", &heapq_replace_cpu, "heapq_replace_cpu");
    m.def("bitmask_put_cpu", &bitmask_put_cpu, "bitmask_put_cpu");
    m.def("square_dist_cpu", &square_dist_cpu, "square_dist_cpu");

    // sampling
    m.def("sampling_cpu", &sampling_cpu, "sampling_cpu");

    // traversal
    m.def("traverse_cpu", &traverse_cpu, "traverse_cpu");
    m.def("traverse_cuda", &traverse_cuda, "traverse_cuda");
    m.def("traverse_refine", &traverse_refine, "traverse_refine");

    // routing
    m.def("routing_cuda", &routing_cuda, "routing_cuda");

    // pipeline
    m.def("pipeline", &pipeline, "pipeline");
}
