// Compact host topology for selected first, diagonal and mixed sensitivities.
#pragma once

#include "common/price_gradients/sensitivity_pair.hpp"
#include "common/price_gradients/sensitivity_request.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <unordered_set>
#include <vector>

namespace ai_factory::workbench::price_gradients {

enum class SensitivityCoordinateUse : std::uint8_t {
    none = 0U,
    first = 1U,
    diagonal_second = 2U,
    mixed_second = 4U,
};

constexpr SensitivityCoordinateUse operator|(
    SensitivityCoordinateUse first,
    SensitivityCoordinateUse second
) {
    return static_cast<SensitivityCoordinateUse>(
        static_cast<std::uint8_t>(first)
        | static_cast<std::uint8_t>(second)
    );
}

__host__ __device__ constexpr bool has_coordinate_use(
    SensitivityCoordinateUse value,
    SensitivityCoordinateUse requested
) {
    return (
        static_cast<std::uint8_t>(value)
        & static_cast<std::uint8_t>(requested)
    ) != 0U;
}

struct SensitivityGraphPlan {
    std::vector<std::uint16_t> first;
    std::vector<std::uint16_t> diagonal_second;
    std::vector<SensitivityPair> mixed_second;
    std::vector<SensitivityCoordinateUse> coordinate_uses;
    std::size_t node_capacity = 1U;

    std::size_t output_count() const noexcept {
        return 1U + first.size() + diagonal_second.size()
            + mixed_second.size();
    }
};

struct DeviceSensitivityGraph {
    const std::uint16_t* first = nullptr;
    std::size_t first_count = 0U;
    std::size_t first_capacity = 0U;
    const std::uint16_t* diagonal_second = nullptr;
    std::size_t diagonal_second_count = 0U;
    std::size_t diagonal_second_capacity = 0U;
    const SensitivityPair* mixed_second = nullptr;
    std::size_t mixed_second_count = 0U;
    std::size_t mixed_second_capacity = 0U;
    const SensitivityCoordinateUse* coordinate_uses = nullptr;
    std::size_t coordinate_use_count = 0U;
    std::size_t coordinate_use_capacity = 0U;
    std::size_t node_capacity = 1U;

    __host__ __device__ std::size_t output_count() const {
        return 1U + first_count + diagonal_second_count
            + mixed_second_count;
    }
};

static_assert(std::is_trivially_copyable_v<DeviceSensitivityGraph>);

namespace sensitivity_graph_plan_detail {

inline void validate_unique_indices(
    const std::vector<std::uint16_t>& indices,
    std::size_t sensitivity_count,
    const char* label
) {
    std::unordered_set<std::uint16_t> unique;
    for (const auto index : indices) {
        if (index >= sensitivity_count || !unique.insert(index).second) {
            throw std::invalid_argument(
                std::string("Invalid or duplicate ") + label
                + " sensitivity index."
            );
        }
    }
}

inline std::vector<std::uint16_t> all_indices(
    std::size_t sensitivity_count
) {
    std::vector<std::uint16_t> result;
    result.reserve(sensitivity_count);
    for (std::size_t index = 0U; index < sensitivity_count; ++index) {
        result.push_back(static_cast<std::uint16_t>(index));
    }
    return result;
}

}  // namespace sensitivity_graph_plan_detail

inline SensitivityGraphPlan make_sensitivity_graph_plan(
    const SensitivityRequest& request,
    std::size_t sensitivity_count
) {
    if (sensitivity_count > std::numeric_limits<std::uint16_t>::max()) {
        throw std::invalid_argument(
            "Sensitivity graph exceeds its compact coordinate index."
        );
    }

    SensitivityGraphPlan result{};
    if (request.has_selected_outputs()) {
        result.first = request.first;
        result.diagonal_second = request.diagonal_second;
        result.mixed_second = request.mixed_second;
    } else {
        if (request.orders == SensitivityOrders::none) {
            throw std::invalid_argument(
                "A sensitivity output must be requested."
            );
        }
        if (request.orders == SensitivityOrders::first
            || request.orders == SensitivityOrders::first_and_second) {
            result.first =
                sensitivity_graph_plan_detail::all_indices(sensitivity_count);
        }
        if (request.orders == SensitivityOrders::second
            || request.orders == SensitivityOrders::first_and_second) {
            result.diagonal_second =
                sensitivity_graph_plan_detail::all_indices(sensitivity_count);
        }
        if (request.all_mixed_second) {
            for (std::size_t first = 0U; first < sensitivity_count; ++first) {
                for (std::size_t second = first + 1U;
                     second < sensitivity_count;
                     ++second) {
                    result.mixed_second.push_back({
                        static_cast<std::uint16_t>(first),
                        static_cast<std::uint16_t>(second),
                    });
                }
            }
        } else {
            result.mixed_second = request.mixed_second;
        }
    }

    sensitivity_graph_plan_detail::validate_unique_indices(
        result.first, sensitivity_count, "first-order"
    );
    sensitivity_graph_plan_detail::validate_unique_indices(
        result.diagonal_second, sensitivity_count, "diagonal-second-order"
    );

    std::unordered_set<std::uint32_t> unique_pairs;
    for (auto& pair : result.mixed_second) {
        if (pair.first >= sensitivity_count
            || pair.second >= sensitivity_count
            || pair.first == pair.second) {
            throw std::invalid_argument(
                "A mixed sensitivity pair is invalid."
            );
        }
        pair = canonical_sensitivity_pair(pair.first, pair.second);
        const std::uint32_t key =
            static_cast<std::uint32_t>(pair.first) << 16U
            | static_cast<std::uint32_t>(pair.second);
        if (!unique_pairs.insert(key).second) {
            throw std::invalid_argument(
                "A mixed sensitivity pair is duplicated."
            );
        }
    }

    result.coordinate_uses.assign(
        sensitivity_count, SensitivityCoordinateUse::none
    );
    const auto add_use = [&](std::uint16_t coordinate,
                             SensitivityCoordinateUse use) {
        result.coordinate_uses[coordinate] =
            result.coordinate_uses[coordinate] | use;
    };
    for (const auto coordinate : result.first) {
        add_use(coordinate, SensitivityCoordinateUse::first);
    }
    for (const auto coordinate : result.diagonal_second) {
        add_use(coordinate, SensitivityCoordinateUse::diagonal_second);
    }
    for (const auto pair : result.mixed_second) {
        add_use(pair.first, SensitivityCoordinateUse::mixed_second);
        add_use(pair.second, SensitivityCoordinateUse::mixed_second);
    }

    std::size_t axial_nodes = 0U;
    for (const auto use : result.coordinate_uses) {
        if (use == SensitivityCoordinateUse::none) continue;
        axial_nodes += has_coordinate_use(
            use, SensitivityCoordinateUse::diagonal_second
        ) ? 3U : 2U;
    }
    if (result.mixed_second.size()
        > (std::numeric_limits<std::size_t>::max() - 1U - axial_nodes) / 4U) {
        throw std::overflow_error("Sensitivity graph node count overflow.");
    }
    result.node_capacity =
        1U + axial_nodes + 4U * result.mixed_second.size();
    return result;
}

}  // namespace ai_factory::workbench::price_gradients
