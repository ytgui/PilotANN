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
std::vector<std::vector<index_t>> graph_sampling_cpu(
    const std::vector<index_t> &indptr, const std::vector<index_t> &indices,
    size_t n_samples, int n_hops
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
    m.def("graph_sampling_cpu", &graph_sampling_cpu, "graph_sampling_cpu");
}
