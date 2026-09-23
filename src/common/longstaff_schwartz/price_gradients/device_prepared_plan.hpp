// Compact sensitivity plan for a fixed early-exercise calendar.
#pragma once

#include "common/equity/price_gradients/device_prepared_plan.hpp"
#include "common/equity/price_gradients/terminal_device_preparation.cuh"

#include <span>

namespace ai_factory::workbench::longstaff_schwartz::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace epg = ::ai_factory::workbench::equity::price_gradients;

template<typename ModelPreparation, typename ProductPreparation>
using DevicePreparedPlan = epg::DevicePreparedSensitivityPlan<
    epg::ScenarioDevicePreparation<
        ModelPreparation,
        ProductPreparation,
        false
    >
>;

template<typename Plan, typename Model, typename Product>
Plan prepare_device_sensitivities(
    std::span<const Model> models,
    std::span<const Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    return epg::prepare_device_sensitivities<Plan>(
        models, products, construction, time, configuration, request
    );
}

}  // namespace ai_factory::workbench::longstaff_schwartz::price_gradients
