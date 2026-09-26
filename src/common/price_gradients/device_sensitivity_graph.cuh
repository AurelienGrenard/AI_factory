// Upload a host sensitivity graph into caller-owned device storage.
#pragma once

#include "common/check_cuda.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::price_gradients {

template<typename Workspace>
DeviceSensitivityGraph upload_sensitivity_graph(
    Workspace workspace,
    const SensitivityGraphPlan& graph
) {
    const auto copy = [](void* destination,
                         const void* source,
                         std::size_t bytes,
                         const char* label) {
        if (bytes == 0U) return;
        check_cuda(
            cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice),
            label
        );
    };
    copy(
        workspace.first_coordinates,
        graph.first.data(),
        graph.first.size() * sizeof(std::uint16_t),
        "sensitivity graph first-coordinate upload"
    );
    copy(
        workspace.diagonal_coordinates,
        graph.diagonal_second.data(),
        graph.diagonal_second.size() * sizeof(std::uint16_t),
        "sensitivity graph diagonal-coordinate upload"
    );
    copy(
        workspace.mixed_pairs,
        graph.mixed_second.data(),
        graph.mixed_second.size() * sizeof(SensitivityPair),
        "sensitivity graph pair upload"
    );
    copy(
        workspace.coordinate_uses,
        graph.coordinate_uses.data(),
        graph.coordinate_uses.size() * sizeof(SensitivityCoordinateUse),
        "sensitivity graph coordinate-use upload"
    );
    return {
        workspace.first_coordinates,
        graph.first.size(),
        workspace.first_coordinate_capacity,
        workspace.diagonal_coordinates,
        graph.diagonal_second.size(),
        workspace.diagonal_coordinate_capacity,
        workspace.mixed_pairs,
        graph.mixed_second.size(),
        workspace.mixed_pair_capacity,
        workspace.coordinate_uses,
        graph.coordinate_uses.size(),
        workspace.coordinate_use_capacity,
        graph.node_capacity,
    };
}

}  // namespace ai_factory::workbench::price_gradients
