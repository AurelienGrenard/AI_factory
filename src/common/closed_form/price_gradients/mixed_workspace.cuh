// Compact caller-owned device storage for one closed-form sensitivity graph.
#pragma once

#include "common/price_gradients/device_prepared_validation.hpp"
#include "common/price_gradients/device_sensitivity_graph.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"
#include "common/workspace_layout.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::closed_form::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

struct MixedWorkspace {
    std::uint16_t* first_coordinates = nullptr;
    std::size_t first_coordinate_capacity = 0U;
    std::uint16_t* diagonal_coordinates = nullptr;
    std::size_t diagonal_coordinate_capacity = 0U;
    pg::SensitivityPair* mixed_pairs = nullptr;
    std::size_t mixed_pair_capacity = 0U;
    pg::SensitivityCoordinateUse* coordinate_uses = nullptr;
    std::size_t coordinate_use_capacity = 0U;
};

struct MixedWorkspaceLayout {
    std::size_t first_coordinates_offset = 0U;
    std::size_t diagonal_coordinates_offset = 0U;
    std::size_t mixed_pairs_offset = 0U;
    std::size_t coordinate_uses_offset = 0U;
    std::size_t first_coordinate_capacity = 0U;
    std::size_t diagonal_coordinate_capacity = 0U;
    std::size_t mixed_pair_capacity = 0U;
    std::size_t coordinate_use_capacity = 0U;
    std::size_t bytes = 0U;
};

inline MixedWorkspaceLayout mixed_workspace_layout(
    const pg::SensitivityGraphPlan& graph
) {
    if (graph.mixed_second.empty()
        || graph.coordinate_uses.empty()
        || graph.node_capacity == 0U) {
        throw std::invalid_argument(
            "The closed-form mixed sensitivity graph is invalid."
        );
    }
    MixedWorkspaceLayout result{};
    result.first_coordinate_capacity = graph.first.size();
    result.diagonal_coordinate_capacity = graph.diagonal_second.size();
    result.mixed_pair_capacity = graph.mixed_second.size();
    result.coordinate_use_capacity = graph.coordinate_uses.size();
    result.first_coordinates_offset =
        ::ai_factory::workbench::workspace_layout::append_array<std::uint16_t>(
            result.bytes, result.first_coordinate_capacity,
            "Closed-form mixed workspace layout overflow."
        );
    result.diagonal_coordinates_offset =
        ::ai_factory::workbench::workspace_layout::append_array<std::uint16_t>(
            result.bytes, result.diagonal_coordinate_capacity,
            "Closed-form mixed workspace layout overflow."
        );
    result.mixed_pairs_offset =
        ::ai_factory::workbench::workspace_layout::append_array<
            pg::SensitivityPair
        >(
            result.bytes, result.mixed_pair_capacity,
            "Closed-form mixed workspace layout overflow."
        );
    result.coordinate_uses_offset =
        ::ai_factory::workbench::workspace_layout::append_array<
            pg::SensitivityCoordinateUse
        >(
            result.bytes, result.coordinate_use_capacity,
            "Closed-form mixed workspace layout overflow."
        );
    return result;
}

inline MixedWorkspace make_mixed_workspace(
    void* storage,
    std::size_t storage_bytes,
    const MixedWorkspaceLayout& layout
) {
    if (storage == nullptr || storage_bytes < layout.bytes) {
        throw std::invalid_argument(
            "Insufficient closed-form mixed byte workspace."
        );
    }
    auto* base = static_cast<unsigned char*>(storage);
    return {
        reinterpret_cast<std::uint16_t*>(
            base + layout.first_coordinates_offset
        ),
        layout.first_coordinate_capacity,
        reinterpret_cast<std::uint16_t*>(
            base + layout.diagonal_coordinates_offset
        ),
        layout.diagonal_coordinate_capacity,
        reinterpret_cast<pg::SensitivityPair*>(
            base + layout.mixed_pairs_offset
        ),
        layout.mixed_pair_capacity,
        reinterpret_cast<pg::SensitivityCoordinateUse*>(
            base + layout.coordinate_uses_offset
        ),
        layout.coordinate_use_capacity,
    };
}

inline void validate_mixed_workspace(
    MixedWorkspace workspace,
    const MixedWorkspaceLayout& required
) {
    const auto validate = [](const void* pointer,
                             std::size_t capacity,
                             std::size_t needed,
                             const char* label) {
        if (capacity < needed) {
            throw std::invalid_argument(
                std::string("Insufficient ") + label + " capacity."
            );
        }
        if (needed != 0U) validate_device_pointer(pointer, label);
    };
    validate(
        workspace.first_coordinates,
        workspace.first_coordinate_capacity,
        required.first_coordinate_capacity,
        "closed-form mixed first coordinates"
    );
    validate(
        workspace.diagonal_coordinates,
        workspace.diagonal_coordinate_capacity,
        required.diagonal_coordinate_capacity,
        "closed-form mixed diagonal coordinates"
    );
    validate(
        workspace.mixed_pairs,
        workspace.mixed_pair_capacity,
        required.mixed_pair_capacity,
        "closed-form mixed coordinate pairs"
    );
    validate(
        workspace.coordinate_uses,
        workspace.coordinate_use_capacity,
        required.coordinate_use_capacity,
        "closed-form mixed coordinate uses"
    );
}

template<typename HostPlan>
std::size_t mixed_workspace_bytes(const HostPlan& host) {
    return mixed_workspace_layout(host.sensitivity_graph).bytes;
}

}  // namespace ai_factory::workbench::closed_form::price_gradients
