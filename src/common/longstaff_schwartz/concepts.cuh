// Compile-time contracts for shared Longstaff-Schwartz execution.
#pragma once

#include "common/longstaff_schwartz/regression_status.cuh"
#include "common/longstaff_schwartz/workspace.cuh"
#include "common/philox.cuh"
#include "common/simulation/concepts.cuh"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::longstaff_schwartz {

template<typename Schedule>
concept EarlyExerciseSchedulePolicy =
    simulation::SchedulePolicy<Schedule>
    && requires(
        const typename Schedule::PreparedSchedule& prepared,
        philox::PhiloxKey key,
        std::size_t path,
        simulation::ObservationHandlerProbe<
            typename Schedule::Dynamics
        >& handler
    ) {
        {
            Schedule::exercise_count(prepared)
        } -> std::same_as<std::uint32_t>;
        {
            Schedule::simulate(prepared, key, path, handler)
        } -> std::same_as<typename Schedule::Dynamics::State>;
    };

template<typename Regressor>
concept SmallLinearRegressor =
    std::is_trivially_copyable_v<typename Regressor::Input>
    && std::is_trivially_copyable_v<typename Regressor::Features>
    && requires(
        const typename Regressor::Input& input,
        const typename Regressor::Features& features,
        double target,
        double (&statistics)[Regressor::kRegressionValueCount],
        std::size_t batch_price,
        std::size_t block_index,
        std::size_t blocks_per_price,
        double* partials,
        double* coefficients,
        RegressionStatus* status,
        RegressionDiagnostics* diagnostics,
        const double* prediction_coefficients,
        std::uint32_t regression_count,
        std::uint32_t backward_level,
        unsigned int threads_per_block
    ) {
        { Regressor::kBasisSize } -> std::convertible_to<std::size_t>;
        {
            Regressor::kRegressionValueCount
        } -> std::convertible_to<std::size_t>;
        {
            Regressor::evaluate(input)
        } -> std::same_as<typename Regressor::Features>;
        {
            Regressor::accumulate(features, target, statistics)
        } -> std::same_as<void>;
        {
            Regressor::reduce_and_store_partials(
                statistics,
                batch_price,
                block_index,
                blocks_per_price,
                partials
            )
        } -> std::same_as<void>;
        {
            Regressor::solve_for_row(
                regression_count,
                backward_level,
                batch_price,
                blocks_per_price,
                partials,
                coefficients,
                status,
                diagnostics
            )
        } -> std::same_as<void>;
        {
            Regressor::predict(features, prediction_coefficients)
        } -> std::same_as<double>;
        {
            Regressor::shared_bytes(threads_per_block)
        } -> std::same_as<std::size_t>;
    };

template<typename PricingPolicy>
concept EarlyExercisePricingPolicy =
    EarlyExerciseSchedulePolicy<typename PricingPolicy::Schedule>
    && std::is_trivially_copyable_v<typename PricingPolicy::DeviceInputs>
    && std::is_trivially_copyable_v<typename PricingPolicy::ProductParameters>
    && std::is_trivially_copyable_v<typename PricingPolicy::PreparedRow>
    && std::is_trivially_copyable_v<typename PricingPolicy::StateView>
    && std::is_trivially_copyable_v<typename PricingPolicy::RegressionInput>
    && requires(
        const typename PricingPolicy::HostInputs& host_inputs,
        const typename PricingPolicy::DeviceInputs& device_inputs,
        const typename PricingPolicy::Schedule::TimeConfiguration&
            time_configuration,
        const typename PricingPolicy::PreparedRow& row,
        const typename PricingPolicy::StateView& states,
        std::size_t result_count,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t paths_per_price,
        std::size_t path,
        std::size_t observation_state_index,
        std::uint32_t backward_level,
        std::uint64_t base_seed,
        float future_cashflow,
        float immediate
    ) {
        { host_inputs.validate(result_count) } -> std::same_as<void>;
        { device_inputs.validate(result_count) } -> std::same_as<void>;
        {
            PricingPolicy::state_field_descriptors()
        } -> std::same_as<std::vector<StateFieldDescriptor>>;
        {
            PricingPolicy::plan_row(
                host_inputs,
                result_index,
                paths_per_price
            )
        } -> std::same_as<EarlyExerciseRowPlan>;
        {
            device_inputs.template prepare_row<PricingPolicy>(
                result_index,
                time_configuration,
                philox::make_key(base_seed + result_index),
                result_index,
                state_offset,
                paths_per_price
            )
        } -> std::same_as<typename PricingPolicy::PreparedRow>;
        {
            PricingPolicy::simulate_path(
                row,
                path,
                paths_per_price,
                states
            )
        } -> std::same_as<float>;
        {
            PricingPolicy::state_index(
                row,
                backward_level,
                paths_per_price,
                path
            )
        } -> std::same_as<std::size_t>;
        {
            PricingPolicy::immediate_value(
                row,
                states,
                observation_state_index
            )
        } -> std::same_as<float>;
        {
            PricingPolicy::regression_input(
                row,
                states,
                observation_state_index
            )
        } -> std::same_as<typename PricingPolicy::RegressionInput>;
        {
            PricingPolicy::regression_candidate(immediate)
        } -> std::same_as<bool>;
        {
            PricingPolicy::regression_target(
                row,
                states,
                observation_state_index,
                future_cashflow
            )
        } -> std::same_as<double>;
        {
            PricingPolicy::continued_cashflow(
                row,
                states,
                observation_state_index,
                future_cashflow
            )
        } -> std::same_as<float>;
        {
            PricingPolicy::initial_continuation_value(
                row,
                states,
                path,
                paths_per_price,
                future_cashflow
            )
        } -> std::same_as<float>;
        {
            PricingPolicy::initial_exercise_value(row)
        } -> std::same_as<float>;
        { row.result_index } -> std::convertible_to<std::size_t>;
        { row.regression_count } -> std::convertible_to<std::uint32_t>;
    };

template<typename PricingPolicy, typename Regressor>
concept LongstaffSchwartzPolicy =
    EarlyExercisePricingPolicy<PricingPolicy>
    && SmallLinearRegressor<Regressor>
    && std::same_as<
        typename PricingPolicy::RegressionInput,
        typename Regressor::Input
    >;

}  // namespace ai_factory::workbench::longstaff_schwartz
