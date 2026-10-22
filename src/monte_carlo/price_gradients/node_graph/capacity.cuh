// Compile-time node capacities for selected sensitivity graphs.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail {

template<std::size_t MaximumSensitivities, std::size_t MaximumMixedSensitivities>
__host__ __device__ constexpr std::size_t mixed_node_graph_node_capacity() {
    static_assert(
        MaximumSensitivities
                <= (0xffffU - 1U - 4U * MaximumMixedSensitivities) / 3U,
        "Mixed node indices exceed their compact representation."
    );
    return 1U + 3U * MaximumSensitivities
        + 4U * MaximumMixedSensitivities;
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients::node_graph_detail
