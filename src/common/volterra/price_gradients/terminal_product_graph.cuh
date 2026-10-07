// Generic terminal-product binding for the host-prepared rough lift graph.
#pragma once

#include "common/equity/price_gradients/terminal_device_prepared_plan.hpp"
#include "common/volterra/price_gradients/prepared_lift_node_graph.cuh"

#include <cstddef>
#include <span>
#include <stdexcept>

namespace ai_factory::workbench::volterra::price_gradients {

namespace epg = ::ai_factory::workbench::equity::price_gradients;

template<typename ModelPreparation, typename ProductPreparation>
using TerminalProductPriceGradientPlan =
    epg::TerminalDevicePreparedPlan<
        ModelPreparation, ProductPreparation
    >;

template<typename ModelPreparation, typename ProductPreparation>
auto prepare_terminal_product_sensitivities(
    std::span<const typename ModelPreparation::Model> models,
    std::span<const typename ProductPreparation::Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_terminal_device_sensitivities<
        TerminalProductPriceGradientPlan<
            ModelPreparation, ProductPreparation
        >
    >(models, products, construction, time, configuration, request);
}

template<pg::SensitivityOrders Orders, typename HostPlan>
void validate_rough_graph_request(const HostPlan& host) {
    if (host.request.orders != Orders
        || !host.sensitivity_graph.mixed_second.empty()) {
        throw std::invalid_argument(
            "This rough node graph requires one dense diagonal "
            "sensitivity order and no mixed derivatives."
        );
    }
    const auto dense = [&](const auto& coordinates) {
        if (coordinates.size() != host.sensitivity_count()) return false;
        for (std::size_t index = 0U; index < coordinates.size(); ++index)
            if (coordinates[index] != index) return false;
        return true;
    };
    if ((pg::requests_first_v<Orders>
            && !dense(host.sensitivity_graph.first))
        || (pg::requests_second_v<Orders>
            && !dense(host.sensitivity_graph.diagonal_second))) {
        throw std::invalid_argument(
            "This rough node graph requires all selected coordinates "
            "in their declared order."
        );
    }
}

template<
    typename Dynamics,
    typename HostPlan,
    typename CachePolicy,
    typename ProductPolicy,
    pg::SensitivityOrders Orders,
    std::size_t MaximumSensitivities>
struct PreparedLiftTerminalProductGraph {
    using Cache = CachePolicy;
    using ExecutionPlan = PreparedLiftGraphExecutionPlan<
        PreparedLiftTerminalNodePolicy<Dynamics, ProductPolicy>
    >;

    static ExecutionPlan plan(
        const HostPlan& host,
        Cache& cache,
        const pg::LaunchConfiguration& launch,
        std::size_t workspace_limit = 1U << 29U
    ) {
        validate_rough_graph_request<Orders>(host);
        return plan_prepared_lift_node_graph_for_host<
            Dynamics, ProductPolicy, Orders, MaximumSensitivities
        >(host, cache, launch, workspace_limit);
    }

    static void launch(
        const HostPlan& host,
        Cache& cache,
        const pg::LaunchConfiguration& configuration,
        pg::SensitivityOutputs outputs,
        mcpg::DevicePreparedStencilOutputs<4U> stencils,
        const ExecutionPlan& execution,
        void* workspace,
        std::size_t workspace_bytes
    ) {
        validate_rough_graph_request<Orders>(host);
        launch_prepared_lift_node_graph<
            Dynamics, ProductPolicy, HostPlan, Cache,
            Orders, MaximumSensitivities
        >(
            host, cache, configuration, outputs, stencils,
            execution, workspace, workspace_bytes
        );
    }
};

}  // namespace ai_factory::workbench::volterra::price_gradients
