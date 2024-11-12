// clang-format off
#include "inc/common.h"
// clang-format on

std::vector<std::vector<index_t>> graph_sampling_cpu(
    const std::vector<index_t> &indptr, const std::vector<index_t> &indices,
    size_t n_samples, int n_hops
) {
    // init
    std::srand(std::time(nullptr));
    auto n_nodes = indptr.size() - 1;

    // expand
    auto expand_neighbors = [&](index_t seed) -> std::set<index_t> {
        auto frontier = std::set<index_t>();

        // loop
        frontier.insert(seed);
        for (auto n = 0; n < n_hops; n += 1) {
            auto next_frontier = std::set<index_t>();
            for (auto node : frontier) {
                auto left = indptr[node];
                auto right = indptr[node + 1];
                for (auto i = left; i < right; i += 1) {
                    auto v = indices[i];
                    if (v < 0) {
                        continue;
                    }
                    next_frontier.insert(v);
                }
            }
            for (auto x : next_frontier) {
                frontier.insert(x);
            }
        }
        return frontier;
    };

    // sample
    auto nodes = std::set<index_t>();
    while (nodes.size() < n_samples) {
        // seed
        auto seed = std::rand() % n_nodes;
        auto neighbors = expand_neighbors(seed);

        // merge
        for (auto x : neighbors) {
            nodes.insert(x);
        }
    }

    // resort
    auto nodelist = std::vector<index_t>(nodes.begin(), nodes.end());
    std::sort(nodelist.begin(), nodelist.end());

    // mapping
    auto mapping = std::vector<index_t>();
    mapping.resize(n_nodes, -1);
    for (auto i = 0; i < nodelist.size(); i += 1) {
        mapping[nodelist[i]] = i;
    }

    //
    return {nodelist, mapping};
}
