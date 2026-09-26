// Compact host plan for row-local sensitivity tasks built on the GPU.
#pragma once

#include "common/price_construction.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/mixed_sensitivity_stencil_outputs.cuh"
#include "common/price_gradients/sensitivity_graph_plan.hpp"
#include "common/price_gradients/sensitivity_outputs.cuh"

#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ai_factory::workbench::price_gradients {

namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename PreparationT>
struct DevicePreparedSensitivityPlan {
    using Preparation = PreparationT;
    using PreparationPolicy = PreparationT;
    using Model = typename Preparation::Model;
    using Product = typename Preparation::Product;
    using SensitivitySpec =
        ::ai_factory::workbench::price_gradients::SensitivitySpec<
            typename Preparation::Parameter
        >;
    using DeviceInputs = mcpg::DevicePreparedInputs<Preparation>;
    using StencilOutputs = mcpg::DevicePreparedStencilOutputs<3U>;
    using DiagonalStencilOutputs = mcpg::DevicePreparedStencilOutputs<4U>;
    using MixedStencilOutputs = MixedSensitivityStencilOutputs;

    static constexpr bool kDevicePreparedSensitivities = true;

    PriceGradientConfiguration configuration;
    SensitivityRequest request;
    TimeConfiguration time;
    PriceConstruction construction;
    std::vector<Model> models;
    std::vector<Product> products;
    std::vector<SensitivitySpec> sensitivities;
    std::size_t result_count;
    SensitivityGraphPlan sensitivity_graph;

    std::size_t sensitivity_count() const noexcept {
        return sensitivities.size();
    }
};

template<typename PreparationT>
struct CurveDevicePreparedSensitivityPlan {
    using Preparation = PreparationT;
    using PreparationPolicy = PreparationT;
    using Model = typename Preparation::Model;
    using Curve = typename Preparation::Curve;
    using Product = typename Preparation::Product;
    using SensitivitySpec =
        ::ai_factory::workbench::price_gradients::SensitivitySpec<
            typename Preparation::Parameter
        >;
    using DeviceInputs = mcpg::CurveDevicePreparedInputs<Preparation>;
    using StencilOutputs = mcpg::DevicePreparedStencilOutputs<3U>;
    using DiagonalStencilOutputs = mcpg::DevicePreparedStencilOutputs<4U>;
    using MixedStencilOutputs = MixedSensitivityStencilOutputs;

    static constexpr bool kDevicePreparedSensitivities = true;

    PriceGradientConfiguration configuration;
    SensitivityRequest request;
    TimeConfiguration time;
    PriceConstruction construction;
    std::vector<Model> models;
    std::vector<Curve> curves;
    std::vector<Product> products;
    std::vector<SensitivitySpec> sensitivities;
    std::size_t result_count;
    SensitivityGraphPlan sensitivity_graph;

    std::size_t sensitivity_count() const noexcept {
        return sensitivities.size();
    }
};

template<typename Plan, typename Model, typename Product>
Plan prepare_device_sensitivities(
    std::span<const Model> models,
    std::span<const Product> products,
    PriceConstruction construction,
    TimeConfiguration time,
    const PriceGradientConfiguration& configuration,
    SensitivityRequest request
) {
    using Preparation = typename Plan::PreparationPolicy;
    time.validate();
    configuration.validate(Preparation::sensitivity_parameter_count);
    if (request.orders == SensitivityOrders::none) {
        throw std::invalid_argument("A sensitivity order must be requested.");
    }
    if (construction != PriceConstruction::Aligned
        && construction != PriceConstruction::CartesianProduct) {
        throw std::invalid_argument("Unknown price construction.");
    }
    const auto rows = price_row_count(
        models.size(), products.size(), construction
    );
    std::vector<typename Plan::SensitivitySpec> sensitivities;
    sensitivities.reserve(configuration.sensitivities.size());
    for (const auto& sensitivity : configuration.sensitivities) {
        sensitivities.push_back({
            Preparation::resolve_parameter(sensitivity.parameter),
            sensitivity.bump,
        });
    }
    auto sensitivity_graph = make_sensitivity_graph_plan(
        request, sensitivities.size()
    );
    for (std::size_t index = 0U; index < sensitivities.size(); ++index) {
        if (has_coordinate_use(
                sensitivity_graph.coordinate_uses[index],
                SensitivityCoordinateUse::diagonal_second
            )
            && Preparation::is_maturity(sensitivities[index].parameter)
            && !Preparation::ModelAdapter::kSupportsMaturityDiagonal) {
            throw std::invalid_argument(
                "Maturity diagonal sensitivity is not supported."
            );
        }
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        const auto model_index = construction == PriceConstruction::Aligned
            ? row
            : row / products.size();
        const auto product_index = construction == PriceConstruction::Aligned
            ? row
            : row % products.size();
        typename Preparation::Scenario central{};
        if (!Preparation::make_central(
                models[model_index], products[product_index], time, central
            )) {
            throw std::invalid_argument(
                "Invalid central sensitivity row " + std::to_string(row)
            );
        }
    }
    return {
        configuration,
        request,
        time,
        construction,
        {models.begin(), models.end()},
        {products.begin(), products.end()},
        std::move(sensitivities),
        rows,
        std::move(sensitivity_graph),
    };
}

template<typename Plan, typename Model, typename Curve, typename Product>
Plan prepare_curve_device_sensitivities(
    std::span<const Model> models,
    std::span<const Curve> curves,
    std::span<const Product> products,
    PriceConstruction construction,
    TimeConfiguration time,
    const PriceGradientConfiguration& configuration,
    SensitivityRequest request
) {
    using Preparation = typename Plan::PreparationPolicy;
    time.validate();
    configuration.validate(Preparation::sensitivity_parameter_count);
    if (request.orders == SensitivityOrders::none) {
        throw std::invalid_argument("A sensitivity order must be requested.");
    }
    if (construction != PriceConstruction::Aligned
        && construction != PriceConstruction::CartesianProduct) {
        throw std::invalid_argument("Unknown price construction.");
    }
    const auto rows = price_row_count(
        models.size(), curves.size(), products.size(), construction
    );
    std::vector<typename Plan::SensitivitySpec> sensitivities;
    sensitivities.reserve(configuration.sensitivities.size());
    for (const auto& sensitivity : configuration.sensitivities) {
        sensitivities.push_back({
            Preparation::resolve_parameter(sensitivity.parameter),
            sensitivity.bump,
        });
    }
    auto sensitivity_graph = make_sensitivity_graph_plan(
        request, sensitivities.size()
    );
    for (std::size_t index = 0U; index < sensitivities.size(); ++index) {
        if (has_coordinate_use(
                sensitivity_graph.coordinate_uses[index],
                SensitivityCoordinateUse::diagonal_second
            )
            && Preparation::is_maturity(sensitivities[index].parameter)
            && !Preparation::ModelAdapter::kSupportsMaturityDiagonal) {
            throw std::invalid_argument(
                "Maturity diagonal sensitivity is not supported."
            );
        }
    }
    for (std::size_t row = 0U; row < rows; ++row) {
        const auto indices = decode_model_curve_product_result_index(
            row, curves.size(), products.size(), construction
        );
        typename Preparation::Scenario central{};
        if (!Preparation::make_central(
                models[indices.model_index],
                curves[indices.curve_index],
                products[indices.product_index],
                time,
                central
            )) {
            throw std::invalid_argument(
                "Invalid central sensitivity row " + std::to_string(row)
            );
        }
    }
    return {
        configuration,
        request,
        time,
        construction,
        {models.begin(), models.end()},
        {curves.begin(), curves.end()},
        {products.begin(), products.end()},
        std::move(sensitivities),
        rows,
        std::move(sensitivity_graph),
    };
}

}  // namespace ai_factory::workbench::price_gradients
