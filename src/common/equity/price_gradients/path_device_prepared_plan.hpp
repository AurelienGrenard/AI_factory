// Host plan for non-exercise path sensitivities built row-locally on the GPU.
#pragma once

#include "common/equity/price_gradients/device_prepared_plan.hpp"
#include "common/equity/price_gradients/terminal_device_preparation.cuh"
#include "common/simulation/calendar.hpp"

#include <span>

namespace ai_factory::workbench::equity::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<typename ModelPreparation, typename ProductPreparation>
using PathDevicePreparedPlan = DevicePreparedSensitivityPlan<
    ScenarioDevicePreparation<
        ModelPreparation,
        ProductPreparation,
        false
    >
>;

template<
    typename Plan,
    bool ExactTransition,
    typename Model,
    typename Product>
Plan prepare_path_device_sensitivities(
    std::span<const Model> models,
    std::span<const Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    using ProductPreparation =
        typename Plan::PreparationPolicy::ProductAdapter;
    for (const auto& product : products) {
        const auto calendar = ProductPreparation::calendar(product);
        if constexpr (ExactTransition) {
            simulation::validate_calendar(
                calendar,
                simulation::ExactTransitionTimeConfiguration{
                    static_cast<float>(time.simulation_steps_per_day) * time.dt
                }
            );
        } else {
            simulation::validate_calendar(
                calendar,
                simulation::FixedStepTimeConfiguration{
                    time.dt, time.simulation_steps_per_day
                }
            );
        }
    }
    return prepare_device_sensitivities<Plan>(
        models,
        products,
        construction,
        time,
        configuration,
        request
    );
}

}  // namespace ai_factory::workbench::equity::price_gradients
