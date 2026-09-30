// Adapt an existing scalar closed-form policy to a prepared sensitivity scenario.
#pragma once

#include "common/simulation/calendar.hpp"

#include <cuda_runtime.h>

#include <type_traits>

namespace ai_factory::workbench::equity::price_gradients {

template<typename BasePolicy>
struct ScenarioClosedFormPolicy {
    template<typename Scenario>
    __device__ __forceinline__ static float evaluate(
        const Scenario& scenario
    ) {
        using TimeConfiguration = typename BasePolicy::TimeConfiguration;
        TimeConfiguration time{};
        if constexpr (std::is_same_v<
                TimeConfiguration,
                simulation::ExactTransitionTimeConfiguration
            >) {
            time.day_fraction = scenario.central_maturity_years
                / static_cast<float>(scenario.product.maturity_days);
        } else {
            time.dt = scenario.central_maturity_years
                / static_cast<float>(scenario.central_step_count);
            time.simulation_steps_per_day =
                scenario.central_step_count
                / scenario.product.maturity_days;
        }
        const auto row = [&] {
            if constexpr (requires {
                BasePolicy::prepare_sensitivity_row(
                    scenario.model,
                    scenario.product,
                    time,
                    scenario.maturity_years
                );
            }) {
                return BasePolicy::prepare_sensitivity_row(
                    scenario.model,
                    scenario.product,
                    time,
                    scenario.maturity_years
                );
            } else {
                TimeConfiguration node_time = time;
                if constexpr (std::is_same_v<
                        TimeConfiguration,
                        simulation::ExactTransitionTimeConfiguration
                    >) {
                    node_time.day_fraction = scenario.maturity_years
                        / static_cast<float>(
                            scenario.product.maturity_days
                        );
                } else {
                    node_time.dt = scenario.maturity_years
                        / static_cast<float>(scenario.step_count);
                    node_time.simulation_steps_per_day =
                        scenario.step_count
                        / scenario.product.maturity_days;
                }
                return BasePolicy::prepare_row(
                    scenario.model, scenario.product, node_time
                );
            }
        }();
        return BasePolicy::evaluate_price(row);
    }
};

}  // namespace ai_factory::workbench::equity::price_gradients
