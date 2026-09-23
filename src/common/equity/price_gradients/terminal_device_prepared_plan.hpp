// Host plan for terminal sensitivities whose row-local nodes are built on GPU.
#pragma once

#include "common/equity/price_gradients/device_prepared_plan.hpp"
#include "common/equity/price_gradients/terminal_device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;

template<typename ModelPreparation, typename ProductPreparation>
using TerminalDevicePreparedPlan = DevicePreparedSensitivityPlan<
    TerminalDevicePreparation<ModelPreparation, ProductPreparation>
>;

template<typename Plan, typename Model, typename Product>
Plan prepare_terminal_device_sensitivities(
    std::span<const Model> models,
    std::span<const Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return prepare_device_sensitivities<Plan>(
        models, products, construction, time, configuration, request
    );
}

}  // namespace ai_factory::workbench::equity::price_gradients
