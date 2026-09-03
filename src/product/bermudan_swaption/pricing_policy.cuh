// Shared Longstaff-Schwartz policy for co-terminal Bermudan swaptions.
#pragma once

#include "common/check_cuda.cuh"
#include "common/device_inputs.cuh"
#include "common/fixed_income/swaption_side.cuh"
#include "common/longstaff_schwartz/workspace.cuh"
#include "common/simulation/early_exercise_schedule.cuh"
#include "product/bermudan_swaption/parameters.hpp"
#include "product/bermudan_swaption/schedule.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

template<typename ProductParameters>
struct EarlyExerciseProductHostInputs {
    const ProductParameters* products;
    std::size_t product_count;
    PriceConstruction construction;

    void validate(std::size_t result_count) const {
        if (products == nullptr) {
            throw std::invalid_argument(
                "Early-exercise host products are null."
            );
        }
        if (product_count == 0U || result_count == 0U) {
            throw std::invalid_argument(
                "Early-exercise host inputs contain an empty dimension."
            );
        }
        if (construction == PriceConstruction::Aligned && product_count != result_count) {
            throw std::invalid_argument(
                "Aligned early-exercise products must match results."
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

template<
    typename SchedulePolicy,
    typename AnalyticsPolicy,
    SwaptionSide Side,
    typename ContinuationStatePolicy,
    typename DeviceInputsPolicy
>
struct BermudanSwaptionPricingPolicyCore {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using Analytics = AnalyticsPolicy;
    using ModelParameters = typename Dynamics::Parameters;
    using ProductParameters = BermudanSwaptionParameters;
    using TimeConfiguration = typename Schedule::TimeConfiguration;
    using DeviceInputs = DeviceInputsPolicy;
    using HostInputs = EarlyExerciseProductHostInputs<ProductParameters>;
    using ContinuationState = ContinuationStatePolicy;
    using StateView = typename ContinuationState::StateView;
    using RegressionInput = typename ContinuationState::RegressionInput;

    static_assert(std::same_as<Dynamics, typename ContinuationState::Dynamics>);

    struct PreparedRow {
        typename Schedule::PreparedSchedule schedule;
        typename Analytics::PreparedModel analytics;
        philox::PhiloxKey key;
        std::size_t result_index;
        std::size_t state_offset;
        std::size_t paths_per_price;
        ProductParameters product;
        [[no_unique_address]] typename Analytics::PreparedRegressionState
            regression_state;
        float time_day_fraction;
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
        std::size_t paths_per_price,
        const TimeConfiguration& time_configuration
    ) {
        const ProductParameters& product = inputs.product(result_index);
        const simulation::RegularExerciseCalendar calendar{
            product.first_exercise_time_days,
            product.payment_interval_days,
            product.exercise_count,
        };
        simulation::validate_exercise_calendar(calendar, time_configuration);
        if (product.payment_count < product.exercise_count) {
            throw std::invalid_argument(
                "Bermudan payment_count must cover every exercise date."
            );
        }
        return {
            product.exercise_count - 1U,
            checked_workspace_product(
                paths_per_price,
                static_cast<std::size_t>(product.exercise_count),
                "Bermudan-swaption row state count exceeds size_t."
            ),
        };
    }

    __device__ __forceinline__ static float exercise_time(
        const PreparedRow& row,
        std::uint32_t exercise
    ) {
        return fmaf(
            static_cast<float>(exercise * row.product.payment_interval_days),
            row.time_day_fraction,
            static_cast<float>(row.product.first_exercise_time_days)
                * row.time_day_fraction
        );
    }

    __device__ __forceinline__ static std::uint32_t exercise_from_state_index(
        const PreparedRow& row,
        std::size_t observation_state_index
    ) {
        return static_cast<std::uint32_t>(
            (observation_state_index - row.state_offset)
                / row.paths_per_price
        );
    }

    template<typename FactorState>
    __device__ __forceinline__ static float immediate_value_at(
        const PreparedRow& row,
        const FactorState& state,
        std::uint32_t exercise
    ) {
        const float time_years = exercise_time(row, exercise);
        const BermudanSwaptionScheduleView schedule =
            make_bermudan_swaption_schedule_view(
                row.product, exercise, row.time_day_fraction
            );
        const float payer_value = Analytics::payer_swap_value(
            row.analytics,
            state,
            time_years,
            time_years,
            row.product.strike,
            schedule
        );
        constexpr float side_sign =
            Side == SwaptionSide::payer ? 1.0f : -1.0f;
        return row.product.notional * fmaxf(side_sign * payer_value, 0.0f);
    }

    __device__ __forceinline__ static PreparedRow prepare_common(
        const ModelParameters& parameters,
        const ProductParameters& product,
        const typename Analytics::PreparedModel& analytics,
        philox::PhiloxKey key,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t paths_per_price,
        const TimeConfiguration& time_configuration
    ) {
        const simulation::RegularExerciseCalendar calendar{
            product.first_exercise_time_days,
            product.payment_interval_days,
            product.exercise_count,
        };
        return {
            Schedule::prepare(parameters, calendar, time_configuration),
            analytics,
            key,
            result_index,
            state_offset,
            paths_per_price,
            product,
            Analytics::prepare_regression_state(parameters),
            simulation::day_count_year_fraction(1U, time_configuration),
            product.exercise_count - 1U,
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
            row.product.exercise_count
        );
        const typename Dynamics::State terminal = Schedule::simulate(
            row.schedule, row.key, path, writer
        );
        return immediate_value_at(
            row,
            Analytics::factor_state(terminal),
            row.product.exercise_count - 1U
        );
    }

    __device__ __forceinline__ static std::size_t state_index(
        const PreparedRow& row,
        std::uint32_t backward_level,
        std::size_t paths_per_price,
        std::size_t path
    ) {
        const std::size_t exercise = static_cast<std::size_t>(
            row.regression_count - 1U - backward_level
        );
        return row.state_offset + exercise * paths_per_price + path;
    }

    __device__ __forceinline__ static float immediate_value(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation_state_index
    ) {
        return immediate_value_at(
            row,
            ContinuationState::factor(states, observation_state_index),
            exercise_from_state_index(row, observation_state_index)
        );
    }

    __device__ __forceinline__ static RegressionInput regression_input(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation_state_index
    ) {
        return ContinuationState::template regression_input<Analytics>(
            row.regression_state, states, observation_state_index
        );
    }

    __device__ __forceinline__ static bool regression_candidate(
        float immediate
    ) {
        return immediate > 0.0f;
    }

    __device__ __forceinline__ static float interval_discount_factor(
        const PreparedRow& row,
        const StateView& states,
        std::size_t current_state_index
    ) {
        const std::uint32_t exercise = exercise_from_state_index(
            row, current_state_index
        );
        const std::size_t next_state_index =
            current_state_index + row.paths_per_price;
        const float current_log_discount = Analytics::log_discount_factor(
            row.analytics,
            ContinuationState::integral(states, current_state_index),
            exercise_time(row, exercise)
        );
        const float next_log_discount = Analytics::log_discount_factor(
            row.analytics,
            ContinuationState::integral(states, next_state_index),
            exercise_time(row, exercise + 1U)
        );
        return expf(next_log_discount - current_log_discount);
    }

    __device__ __forceinline__ static double regression_target(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation_state_index,
        float future_cashflow
    ) {
        return static_cast<double>(interval_discount_factor(
            row, states, observation_state_index
        )) * static_cast<double>(future_cashflow);
    }

    __device__ __forceinline__ static float continued_cashflow(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation_state_index,
        float future_cashflow
    ) {
        return interval_discount_factor(
            row, states, observation_state_index
        ) * future_cashflow;
    }

    __device__ __forceinline__ static float initial_continuation_value(
        const PreparedRow& row,
        const StateView& states,
        std::size_t path,
        std::size_t,
        float cashflow
    ) {
        const std::size_t first_state_index = row.state_offset + path;
        const float log_discount = Analytics::log_discount_factor(
            row.analytics,
            ContinuationState::integral(states, first_state_index),
            exercise_time(row, 0U)
        );
        return expf(log_discount) * cashflow;
    }

    __device__ __forceinline__ static float initial_exercise_value(
        const PreparedRow&
    ) {
        return 0.0f;
    }
};

template<
    typename SchedulePolicy,
    typename AnalyticsPolicy,
    SwaptionSide Side,
    typename ContinuationStatePolicy
>
struct StandaloneBermudanSwaptionPricingPolicy
    : BermudanSwaptionPricingPolicyCore<
        SchedulePolicy,
        AnalyticsPolicy,
        Side,
        ContinuationStatePolicy,
        ModelProductDeviceInputs<
            typename SchedulePolicy::Dynamics::Parameters,
            BermudanSwaptionParameters
        >
    > {
    using Base = BermudanSwaptionPricingPolicyCore<
        SchedulePolicy,
        AnalyticsPolicy,
        Side,
        ContinuationStatePolicy,
        ModelProductDeviceInputs<
            typename SchedulePolicy::Dynamics::Parameters,
            BermudanSwaptionParameters
        >
    >;
    using typename Base::ModelParameters;
    using typename Base::PreparedRow;
    using typename Base::ProductParameters;
    using typename Base::TimeConfiguration;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        philox::PhiloxKey key,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t paths_per_price,
        const TimeConfiguration& time_configuration
    ) {
        return Base::prepare_common(
            model,
            product,
            AnalyticsPolicy::prepare_model(model),
            key,
            result_index,
            state_offset,
            paths_per_price,
            time_configuration
        );
    }
};

template<
    typename SchedulePolicy,
    typename CurveParameters,
    typename AnalyticsPolicy,
    SwaptionSide Side,
    typename ContinuationStatePolicy
>
struct FittedBermudanSwaptionPricingPolicy
    : BermudanSwaptionPricingPolicyCore<
        SchedulePolicy,
        AnalyticsPolicy,
        Side,
        ContinuationStatePolicy,
        ModelCurveProductDeviceInputs<
            typename SchedulePolicy::Dynamics::Parameters,
            CurveParameters,
            BermudanSwaptionParameters
        >
    > {
    using Base = BermudanSwaptionPricingPolicyCore<
        SchedulePolicy,
        AnalyticsPolicy,
        Side,
        ContinuationStatePolicy,
        ModelCurveProductDeviceInputs<
            typename SchedulePolicy::Dynamics::Parameters,
            CurveParameters,
            BermudanSwaptionParameters
        >
    >;
    using typename Base::ModelParameters;
    using typename Base::PreparedRow;
    using typename Base::ProductParameters;
    using typename Base::TimeConfiguration;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const CurveParameters& curve,
        const ProductParameters& product,
        philox::PhiloxKey key,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t paths_per_price,
        const TimeConfiguration& time_configuration
    ) {
        return Base::prepare_common(
            model,
            product,
            AnalyticsPolicy::prepare_model(model, curve),
            key,
            result_index,
            state_offset,
            paths_per_price,
            time_configuration
        );
    }
};

}  // namespace ai_factory::workbench::product
