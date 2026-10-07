// Reuse common sensitivity stencils while mapping lift nodes to cached dynamics.
#pragma once

#include "common/monte_carlo/price_gradients/node_graph/row_preparation.cuh"
#include "common/result_index.cuh"
#include "common/price_gradients/device_prepared_plan.hpp"
#include "common/volterra/price_gradients/prepared_lift_node_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::volterra::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename Scenario, std::size_t MaximumSensitivities>
struct PreparedLiftGraphRow {
    static constexpr std::size_t kNodeCapacity =
        mcpg::terminal_node_graph_node_capacity<MaximumSensitivities>();
    Scenario scenarios[kNodeCapacity]{};
    std::uint32_t prepared_indices[kNodeCapacity]{};
    pg::SensitivityStencil<4U> stencils[MaximumSensitivities]{};
    mcpg::SensitivityNodeIndices<4U>
        node_indices[MaximumSensitivities]{};
    std::uint16_t node_count = 0U;
    std::uint32_t maximum_steps = 0U;
};

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename HostPlan>
PreparedLiftGraphRow<
    typename HostPlan::Preparation::Scenario,
    MaximumSensitivities
> prepare_lift_graph_row_scenarios(
    const HostPlan& host,
    std::size_t row
) {
    static_assert(Orders != pg::SensitivityOrders::none);
    using Preparation = typename HostPlan::Preparation;
    if (row >= host.result_count
        || host.sensitivity_count() == 0U
        || host.sensitivity_count() > MaximumSensitivities) {
        throw std::invalid_argument("Invalid prepared-lift graph row.");
    }
    const auto indices = decode_model_product_result_index(
        row,
        host.products.size(),
        host.construction
    );
    typename Preparation::Scenario central{};
    if (!Preparation::make_central(
            host.models[indices.model_index],
            host.products[indices.product_index],
            host.time,
            central
        )) {
        throw std::invalid_argument("Invalid prepared-lift central row.");
    }

    PreparedLiftGraphRow<
        typename Preparation::Scenario, MaximumSensitivities
    > result{};
    int error = pg::device_preparation::valid;
    std::size_t error_sensitivity = 0U;
    if (!mcpg::node_graph_detail::prepare_sensitivity_row_from_central<
            Orders, Preparation, MaximumSensitivities
        >(
            central,
            host.sensitivities.data(),
            host.sensitivity_count(),
            host.time,
            result.scenarios,
            result.stencils,
            result.node_indices,
            result.node_count,
            result.maximum_steps,
            error,
            error_sensitivity
        )) {
        throw std::invalid_argument(
            "Invalid prepared-lift sensitivity node "
            + std::to_string(error_sensitivity)
            + " (error " + std::to_string(error) + ")."
        );
    }
    return result;
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename HostPlan,
    typename Cache>
auto prepare_lift_graph_row(
    const HostPlan& host,
    std::size_t row,
    Cache& cache
) {
    auto result = prepare_lift_graph_row_scenarios<
        Orders, MaximumSensitivities
    >(host, row);
    for (std::size_t node = 0U; node < result.node_count; ++node) {
        auto model = result.scenarios[node].model;
        // Spot-only bumps rescale the central simulation. Product-only bumps
        // retain the same dynamics. Both hit the same cache entry.
        model.spot = result.scenarios[node].simulation_spot;
        result.prepared_indices[node] = cache.index(model);
    }
    return result;
}

template<
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities,
    typename HostPlan,
    typename Cache>
void prepare_lift_model_node_cache(
    const HostPlan& host,
    Cache& cache
) {
    // Every product-only node retains its model. Model bump endpoints are
    // independent of the product in a Cartesian request. Fit their distinct
    // H values in parallel, then prepare complete dynamics in row order.
    std::vector<float> hurst_values;
    hurst_values.reserve(host.models.size() * (
        1U + 3U * host.sensitivity_count()
    ));
    for (std::size_t model = 0U; model < host.models.size(); ++model) {
        const auto row = host.construction == PriceConstruction::Aligned
            ? model
            : model * host.products.size();
        const auto scenarios = prepare_lift_graph_row_scenarios<
            Orders, MaximumSensitivities
        >(host, row);
        for (std::size_t node = 0U; node < scenarios.node_count; ++node) {
            auto scenario_model = scenarios.scenarios[node].model;
            scenario_model.spot = scenarios.scenarios[node].simulation_spot;
            if (!cache.contains_prepared(scenario_model))
                hurst_values.push_back(scenario_model.hurst_exponent);
        }
    }
    cache.prefit(hurst_values);
    for (std::size_t model = 0U; model < host.models.size(); ++model) {
        const auto row = host.construction == PriceConstruction::Aligned
            ? model
            : model * host.products.size();
        (void)prepare_lift_graph_row<Orders, MaximumSensitivities>(
            host, row, cache
        );
    }
}

}  // namespace ai_factory::workbench::volterra::price_gradients
