// Generic American-option Longstaff-Schwartz pricing policy.
#pragma once

#include "common/check_cuda.cuh"
#include "common/device_inputs.cuh"
#include "common/equity/discount.cuh"
#include "common/option_side.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "common/simulation/early_exercise_schedule.cuh"
#include "product/american_option/parameters.hpp"

#include <cuda_runtime.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

template<
    typename SchedulePolicy,
    OptionSide Side,
    typename ContinuationStatePolicy
>
struct AmericanOptionPricingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = AmericanOptionParameters;
    using TimeConfiguration = typename Schedule::TimeConfiguration;
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        ProductParameters
    >;
    using ContinuationState = ContinuationStatePolicy;
    using StateView = typename ContinuationState::StateView;
    using RegressionInput = typename ContinuationState::RegressionInput;

    static_assert(std::same_as<
        Dynamics,
        typename ContinuationState::Dynamics
    >);

    struct HostInputs {
        const ProductParameters* products;
        std::size_t product_count;
        PriceConstruction construction;

        void validate(std::size_t result_count) const {
            if (products == nullptr) {
                throw std::invalid_argument(
                    "American-option host products are null."
                );
            }
            if (product_count == 0U || result_count == 0U) {
                throw std::invalid_argument(
                    "American-option host inputs contain an empty dimension."
                );
            }
            if (construction == PriceConstruction::Aligned && product_count != result_count) {
                throw std::invalid_argument(
                    "Aligned American-option host products must match results."
                );
            }
        }

        const ProductParameters& product(std::size_t result_index) const {
            const std::size_t product_index = is_cartesian(construction)
                ? result_index % product_count
                : result_index;
            return products[product_index];
        }
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        philox::PhiloxKey key;
        std::size_t result_index;
        std::size_t state_offset;
        float strike;
        float initial_spot;
        float inverse_strike;
        [[no_unique_address]] typename ContinuationState::PreparedState
            continuation_state;
        float exercise_discount;
        float initial_discount;
        std::uint32_t regression_count;
    };

    static_assert(std::is_trivially_copyable_v<PreparedRow>);
    static_assert(std::is_trivially_copyable_v<StateView>);

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    state_field_descriptors() {
        return ContinuationState::state_field_descriptors();
    }

    static StateView make_state_view(
        unsigned char* workspace,
        const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        return ContinuationState::make_state_view(workspace, layout);
    }

    static longstaff_schwartz::EarlyExerciseRowPlan plan_row(
        const HostInputs& inputs,
        std::size_t result_index,
        std::size_t paths_per_price
    ) {
        const ProductParameters& product = inputs.product(result_index);
        const simulation::MaturityAlignedExerciseCalendar calendar{
            product.maturity_days,
            product.exercise_interval_days,
        };
        simulation::validate_exercise_calendar(calendar);
        const std::uint32_t exercise_count =
            simulation::maturity_aligned_exercise_count(calendar);
        const std::uint32_t regression_count = exercise_count - 1U;
        return {
            regression_count,
            checked_workspace_product(
                paths_per_price,
                static_cast<std::size_t>(regression_count),
                "American-option row state count exceeds size_t."
            ),
        };
    }

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& parameters,
        const ProductParameters& product,
        philox::PhiloxKey key,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t,
        const TimeConfiguration& time_configuration
    ) {
        const simulation::MaturityAlignedExerciseCalendar calendar{
            product.maturity_days,
            product.exercise_interval_days,
        };
        const std::uint32_t first_exercise_days =
            simulation::maturity_aligned_first_exercise_days(calendar);
        const float exercise_interval = simulation::day_count_year_fraction(
            product.exercise_interval_days,
            time_configuration
        );
        const float first_exercise_time = simulation::day_count_year_fraction(
            first_exercise_days,
            time_configuration
        );
        const typename Schedule::PreparedSchedule schedule =
            Schedule::prepare(parameters, calendar, time_configuration);
        return {
            schedule,
            key,
            result_index,
            state_offset,
            product.strike,
            static_cast<float>(parameters.spot),
            1.0f / product.strike,
            ContinuationState::prepare(parameters),
            equity::constant_rate_discount_factor(
                parameters, exercise_interval
            ),
            equity::constant_rate_discount_factor(
                parameters, first_exercise_time
            ),
            Schedule::exercise_count(schedule) - 1U,
        };
    }

    __device__ __forceinline__ static float simulate_path(
        const PreparedRow& row,
        std::size_t path,
        std::size_t paths_per_price,
        const StateView& states
    ) {
        auto writer = ContinuationState::make_writer(
            states,
            row.state_offset + path,
            paths_per_price,
            row.regression_count
        );
        const typename Dynamics::State terminal = Schedule::simulate(
            row.schedule,
            row.key,
            path,
            writer
        );
        return payoff::vanilla_option_payoff<Side>(
            Dynamics::spot(terminal), row.strike
        );
    }

    __device__ __forceinline__ static std::size_t state_index(
        const PreparedRow& row,
        std::uint32_t backward_level,
        std::size_t paths_per_price,
        std::size_t path
    ) {
        const std::size_t regression_exercise = static_cast<std::size_t>(
            row.regression_count - 1U - backward_level
        );
        return row.state_offset
            + regression_exercise * paths_per_price
            + path;
    }

    __device__ __forceinline__ static float immediate_value(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation_state_index
    ) {
        return payoff::vanilla_option_payoff<Side>(
            ContinuationState::spot(states, observation_state_index),
            row.strike
        );
    }

    __device__ __forceinline__ static RegressionInput regression_input(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation_state_index
    ) {
        return ContinuationState::regression_input(
            row.continuation_state,
            states,
            observation_state_index,
            row.inverse_strike
        );
    }

    __device__ __forceinline__ static bool regression_candidate(
        float immediate
    ) {
        return immediate > 0.0f;
    }

    __device__ __forceinline__ static double regression_target(
        const PreparedRow& row,
        const StateView&,
        std::size_t,
        float future_cashflow
    ) {
        return static_cast<double>(row.exercise_discount)
            * static_cast<double>(future_cashflow);
    }

    __device__ __forceinline__ static float continued_cashflow(
        const PreparedRow& row,
        const StateView&,
        std::size_t,
        float future_cashflow
    ) {
        return row.exercise_discount * future_cashflow;
    }

    __device__ __forceinline__ static float initial_continuation_value(
        const PreparedRow& row,
        const StateView&,
        std::size_t,
        std::size_t,
        float cashflow
    ) {
        return row.initial_discount * cashflow;
    }

    __device__ __forceinline__ static float initial_exercise_value(
        const PreparedRow& row
    ) {
        return payoff::vanilla_option_payoff<Side>(
            row.initial_spot, row.strike
        );
    }
};

}  // namespace ai_factory::workbench::product
