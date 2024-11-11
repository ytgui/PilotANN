// clang-format off
#include "inc/common.h"
#include "inc/heapq.cuh"
#include "inc/bitmask.hpp"
#include "inc/distance.h"
// clang-format on

void heapq_pop_cpu(torch::Tensor &keys, torch::Tensor &values) {
    CHECK_CPU(keys, 1, torch::kFloat32);
    CHECK_CPU(values, 1, torch::kInt64);
    TORCH_CHECK(keys.numel() == values.numel());
    heap_pop(keys.data_ptr<float>(), values.data_ptr<index_t>(), keys.numel());
}

void heapq_pushpop_cpu(
    torch::Tensor &keys, torch::Tensor &values, float k, index_t v
) {
    CHECK_CPU(keys, 1, torch::kFloat32);
    CHECK_CPU(values, 1, torch::kInt64);
    TORCH_CHECK(keys.numel() == values.numel());
    heap_pushpop(
        keys.data_ptr<float>(), values.data_ptr<index_t>(), keys.numel(), k, v
    );
}

void heapq_replace_cpu(
    torch::Tensor &keys, torch::Tensor &values, float k, index_t v
) {
    CHECK_CPU(keys, 1, torch::kFloat32);
    CHECK_CPU(values, 1, torch::kInt64);
    TORCH_CHECK(keys.numel() == values.numel());
    heap_replace(
        keys.data_ptr<float>(), values.data_ptr<index_t>(), keys.numel(), k, v
    );
}

torch::Tensor bitmask_put_cpu(const torch::Tensor &keys, index_t n) {
    CHECK_CPU(keys, 1, torch::kInt64);
    auto bitmask = Bitmask<int32_t>(n, 1, keys.device());

    // simple
    bitmask.advance();
    auto mask_value = bitmask.marker();
    auto bitmask_ptr = bitmask.data_ptr();
    auto keys_accessor = keys.accessor<index_t, 1>();
    for (auto i = 0; i < keys.size(0); i += 1) {
        auto x = keys_accessor[i];
        bitmask_ptr[x] = mask_value;
    }
    return bitmask.tensor();
}

torch::Tensor square_dist_cpu(
    const torch::Tensor &query, const torch::Tensor &storage,
    const torch::Tensor &nodelist
) {
    CHECK_CPU(query, 1, torch::kFloat32);
    CHECK_CPU(storage, 2, torch::kFloat32);
    CHECK_CPU(nodelist, 1, torch::kInt64);

    // sizes
    auto d_model = query.size(0);
    auto n_nodes = nodelist.size(0);
    TORCH_CHECK(d_model % SIMD_WIDTH == 0);
    TORCH_CHECK(storage.size(-1) == d_model);
    auto output = torch::zeros({n_nodes}, query.options());

    // compute
    auto cursor = 0;
    const auto tile_size = 4;
    while (cursor + tile_size < n_nodes) {
        compute_dist_avx2c<tile_size>(
            output.data_ptr<float>() + cursor, query.data_ptr<float>(),
            storage.data_ptr<float>(), nodelist.data_ptr<index_t>() + cursor,
            d_model
        );
        cursor += tile_size;
    }
    compute_dist_avx2r(
        output.data_ptr<float>() + cursor, query.data_ptr<float>(),
        storage.data_ptr<float>(), nodelist.data_ptr<index_t>() + cursor,
        d_model, n_nodes - cursor
    );
    return output;
}
