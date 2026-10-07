// Verify mixed sensitivity stencil reconstruction and graph contracts.
#include "common/equity/price_gradients/terminal_device_preparation.cuh"
#include "common/monte_carlo/price_gradients/node_graph/mixed_row_preparation.cuh"
#include "common/price_gradients/mixed_sensitivity_reconstruction.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"
#include "model/equity/markovian/heston/price_gradients/device_preparation.cuh"
#include "product/european_option/price_gradients/device_preparation.cuh"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

namespace pg = ai_factory::workbench::price_gradients;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<typename Callable>
void rejects(Callable&& callable) {
    bool rejected = false;
    try {
        callable();
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "Invalid mixed sensitivity request was accepted.");
}

pg::SensitivityStencil<4U> centered(float central, float h) {
    pg::SensitivityStencil<4U> result{};
    result.kind = pg::StencilKind::centered;
    result.parameter_values[0U] = central;
    result.parameter_values[1U] = central - h;
    result.parameter_values[2U] = central + h;
    result.represented_width = 2.0f * h;
    result.first_endpoint_weights[0U] = -1.0f / result.represented_width;
    result.first_endpoint_weights[1U] = 1.0f / result.represented_width;
    result.node_count = 3U;
    return result;
}

pg::SensitivityStencil<4U> forward(float central, float h) {
    pg::SensitivityStencil<4U> result{};
    result.kind = pg::StencilKind::forward;
    result.parameter_values[0U] = central;
    result.parameter_values[1U] = central + h;
    result.parameter_values[2U] = central + 2.0f * h;
    result.parameter_values[3U] = central + 3.0f * h;
    result.first_endpoint_weights[0U] = 2.0f / h;
    result.first_endpoint_weights[1U] = -0.5f / h;
    result.node_count = 4U;
    return result;
}

float function(float x, float y) {
    return 2.0f * x * y + 3.0f * x * x + 4.0f * y + 7.0f;
}

void check_reconstruction(
    const pg::SensitivityStencil<4U>& first,
    const pg::SensitivityStencil<4U>& second,
    std::size_t expected_nodes
) {
    const auto stencil = pg::make_mixed_sensitivity_stencil(first, second);
    require(stencil.node_count == expected_nodes,
            "Mixed tensor-product support has the wrong size.");
    pg::MixedSensitivityValues values{};
    float weight_sum = 0.0f;
    for (std::size_t node = 0U; node < stencil.node_count; ++node) {
        const float x = first.parameter_values[
            stencil.first_local_nodes[node]
        ];
        const float y = second.parameter_values[
            stencil.second_local_nodes[node]
        ];
        values[node] = function(x, y);
        weight_sum += stencil.weights[node];
    }
    require(std::abs(weight_sum) < 2.0e-4f,
            "Mixed finite-difference weights do not annihilate constants.");
    const float central = function(
        first.parameter_values[0U], second.parameter_values[0U]
    );
    require(
        std::abs(
            pg::reconstruct_mixed_sensitivity(stencil, values, central)
            - 2.0f
        ) < 2.0e-3f,
        "Mixed tensor-product reconstruction is incorrect."
    );
}

void prepared_heston_row() {
    using ModelPreparation =
        ai_factory::workbench::model::equity::heston::price_gradients::
            DevicePreparation;
    using ProductPreparation =
        ai_factory::workbench::product::european_option::price_gradients::
            DevicePreparation;
    using Preparation =
        ai_factory::workbench::equity::price_gradients::
            TerminalDevicePreparation<ModelPreparation, ProductPreparation>;
    namespace mcpg =
        ai_factory::workbench::monte_carlo::price_gradients;
    namespace detail = mcpg::node_graph_detail;

    constexpr std::size_t sensitivity_capacity = 3U;
    constexpr std::size_t mixed_capacity = 3U;
    constexpr std::size_t node_capacity =
        detail::mixed_node_graph_node_capacity<
            sensitivity_capacity, mixed_capacity
        >();
    typename Preparation::Scenario central{};
    const typename Preparation::Model model{
        1.0f, 0.0f, 0.0f, 0.04f, 1.5f, 0.04f, 0.3f, -0.7f
    };
    const typename Preparation::Product product{1.0f, 16U};
    const pg::TimeConfiguration time{};
    require(
        Preparation::make_central(model, product, time, central),
        "Heston mixed central row is invalid."
    );

    const pg::SensitivitySpec<typename Preparation::Parameter> sensitivities[]{
        {
            Preparation::resolve_parameter("model.initial_variance"),
            {0.001f, pg::BumpScale::absolute},
        },
        {
            Preparation::resolve_parameter("model.rho"),
            {0.002f, pg::BumpScale::absolute},
        },
        {
            Preparation::resolve_parameter("product.strike"),
            {0.005f},
        },
    };
    const auto host_graph = pg::make_sensitivity_graph_plan(
        pg::SensitivityRequest::full_hessian(), sensitivity_capacity
    );
    const pg::DeviceSensitivityGraph graph{
        host_graph.first.data(),
        host_graph.first.size(),
        host_graph.first.size(),
        host_graph.diagonal_second.data(),
        host_graph.diagonal_second.size(),
        host_graph.diagonal_second.size(),
        host_graph.mixed_second.data(),
        host_graph.mixed_second.size(),
        host_graph.mixed_second.size(),
        host_graph.coordinate_uses.data(),
        host_graph.coordinate_uses.size(),
        host_graph.coordinate_uses.size(),
        host_graph.node_capacity,
    };
    typename Preparation::Scenario scenarios[node_capacity]{};
    pg::SensitivityStencil<4U> stencils[sensitivity_capacity]{};
    mcpg::SensitivityNodeIndices<4U> axis_indices[sensitivity_capacity]{};
    pg::MixedSensitivityStencil mixed_stencils[mixed_capacity]{};
    mcpg::MixedSensitivityNodeIndices mixed_indices[mixed_capacity]{};
    std::uint16_t node_count = 0U;
    std::uint32_t maximum_steps = 0U;
    int error = pg::device_preparation::valid;
    std::size_t error_sensitivity = 0U;
    require(
        detail::prepare_mixed_sensitivity_row_from_central<
            Preparation, sensitivity_capacity, mixed_capacity
        >(
            central,
            sensitivities,
            sensitivity_capacity,
            graph,
            time,
            scenarios,
            stencils,
            axis_indices,
            mixed_stencils,
            mixed_indices,
            node_count,
            maximum_steps,
            error,
            error_sensitivity
        ),
        "Heston mixed row preparation failed."
    );
    require(node_count == 19U && maximum_steps == central.step_count,
            "Heston centered full-Hessian node count is incorrect.");
    require(mixed_stencils[1U].node_count == 4U,
            "Heston centered mixed stencil is not four-corner.");

    const auto variance_strike = host_graph.mixed_second[1U];
    require(variance_strike == pg::SensitivityPair{0U, 2U},
            "Unexpected Heston mixed-pair ordering.");
    for (std::size_t local = 0U; local < 4U; ++local) {
        const auto node = mixed_indices[1U][local];
        const auto first_local = mixed_stencils[1U].first_local_nodes[local];
        const auto second_local = mixed_stencils[1U].second_local_nodes[local];
        require(
            scenarios[node].model.initial_variance
                == stencils[0U].parameter_values[first_local]
                && scenarios[node].product.strike
                    == stencils[2U].parameter_values[second_local],
            "A Heston mixed corner did not apply both parameter endpoints."
        );
    }
}

void graph_plan() {
    const auto legacy = pg::make_sensitivity_graph_plan(
        pg::SensitivityRequest{pg::SensitivityOrders::first_and_second}, 3U
    );
    require(legacy.first.size() == 3U
            && legacy.diagonal_second.size() == 3U
            && legacy.mixed_second.empty()
            && legacy.node_capacity == 10U
            && legacy.output_count() == 7U,
            "Legacy gradient/diagonal graph shape changed.");

    const auto full = pg::make_sensitivity_graph_plan(
        pg::SensitivityRequest::full_hessian(), 3U
    );
    require(full.mixed_second.size() == 3U
            && full.node_capacity == 22U
            && full.output_count() == 10U,
            "Full-Hessian graph shape is incorrect.");
    require(full.mixed_second[0U] == pg::SensitivityPair{0U, 1U}
            && full.mixed_second[1U] == pg::SensitivityPair{0U, 2U}
            && full.mixed_second[2U] == pg::SensitivityPair{1U, 2U},
            "Full-Hessian pair ordering is not canonical.");

    const auto sparse = pg::make_sensitivity_graph_plan(
        pg::SensitivityRequest::selected(
            {2U}, {0U}, {{2U, 1U}}
        ),
        3U
    );
    require(sparse.first == std::vector<std::uint16_t>{2U}
            && sparse.diagonal_second == std::vector<std::uint16_t>{0U}
            && sparse.mixed_second[0U] == pg::SensitivityPair{1U, 2U}
            && sparse.node_capacity == 12U
            && sparse.output_count() == 4U,
            "Sparse mixed-sensitivity graph shape is incorrect.");
    require(pg::has_coordinate_use(
                sparse.coordinate_uses[0U],
                pg::SensitivityCoordinateUse::diagonal_second
            )
            && pg::has_coordinate_use(
                sparse.coordinate_uses[1U],
                pg::SensitivityCoordinateUse::mixed_second
            )
            && pg::has_coordinate_use(
                sparse.coordinate_uses[2U],
                pg::SensitivityCoordinateUse::first
            ),
            "Sparse coordinate requirements are incorrect.");

    rejects([] {
        pg::make_sensitivity_graph_plan(
            pg::SensitivityRequest::selected({}, {}, {{0U, 0U}}), 2U
        );
    });
    rejects([] {
        pg::make_sensitivity_graph_plan(
            pg::SensitivityRequest::selected(
                {}, {}, {{0U, 1U}, {1U, 0U}}
            ),
            2U
        );
    });
    rejects([] {
        pg::make_sensitivity_graph_plan(
            pg::SensitivityRequest::selected({2U}, {}, {}), 2U
        );
    });
}

}  // namespace

int main() {
    try {
        graph_plan();
        prepared_heston_row();
        check_reconstruction(centered(1.0f, 0.1f), centered(2.0f, 0.2f), 4U);
        check_reconstruction(centered(1.0f, 0.1f), forward(2.0f, 0.2f), 6U);
        check_reconstruction(forward(1.0f, 0.1f), forward(2.0f, 0.2f), 9U);
        std::cout << "Mixed sensitivity request, graph plan and reconstruction pass\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
