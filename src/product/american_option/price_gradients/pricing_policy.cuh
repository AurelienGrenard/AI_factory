// American selected gradients with one central LSM solve and frozen exercise.
#pragma once

#include "common/equity/price_gradients/scenarios.hpp"
#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/longstaff_schwartz/price_gradients/frozen_exercise_kernels.cuh"
#include "common/longstaff_schwartz/price_gradients/workspace.cuh"
#include "common/price_gradients/launch.cuh"
#include "product/american_option/price_gradients/frozen_exercise_evaluator.cuh"
#include "product/american_option/frozen_exercise_value.cuh"
#include "product/american_option/pricing_policy.cuh"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace ai_factory::workbench::product {

template<
    typename SchedulePolicy,
    OptionSide Side,
    typename Continuation,
    typename ReplayPolicy
>
struct AmericanOptionPriceGradientPolicy
    : AmericanOptionPricingPolicy<SchedulePolicy, Side, Continuation> {
    using Base = AmericanOptionPricingPolicy<
        SchedulePolicy, Side, Continuation
    >;
    using Schedule = SchedulePolicy;
    using ModelParameters = typename Base::ModelParameters;
    using ProductParameters = typename Base::ProductParameters;
    using TimeConfiguration = typename Base::TimeConfiguration;
    using Scenario = ::ai_factory::workbench::equity::price_gradients::Scenario<
        ModelParameters, ProductParameters
    >;
    using Evaluator = american_option::price_gradients::FrozenExerciseEvaluator<
        Side, ReplayPolicy
    >;
    using PreparedSensitivity =
        typename Evaluator::template Prepared<Scenario>;
    using Exercise = longstaff_schwartz::FrozenExerciseTrace;
    using InitialDecision = longstaff_schwartz::InitialExerciseDecision;

    struct HostInputs {
        const Scenario* scenarios;
        std::size_t scenario_count;
        const ::ai_factory::workbench::price_gradients::Stencil* stencils;
        std::size_t stencil_count;
        std::size_t sensitivity_count;

        void validate(std::size_t results) const {
            if (scenarios == nullptr || results == 0U) {
                throw std::invalid_argument(
                    "American gradient host scenarios are null or empty."
                );
            }
            const std::size_t scenarios_per_row =
                1U + 2U * sensitivity_count;
            if (scenario_count != results * scenarios_per_row
                || stencil_count != results * sensitivity_count
                || (sensitivity_count != 0U && stencils == nullptr)) {
                throw std::invalid_argument(
                    "Malformed American gradient host scenario plan."
                );
            }
        }

        const ProductParameters& product(std::size_t result_index) const {
            return scenarios[
                result_index * (1U + 2U * sensitivity_count)
            ].product;
        }
    };

    struct DeviceInputs {
        ::ai_factory::workbench::price_gradients::DeviceInputs<Scenario>
            primary;
        std::size_t sensitivity_count;
        std::size_t output_result_offset;
        float* gradients;
        float* gradient_errors;

        void validate(std::size_t results) const {
            validate_device_pointer(primary.scenarios, "gradient scenarios");
            const std::size_t scenarios_per_row =
                1U + 2U * sensitivity_count;
            if (primary.scenario_capacity < results * scenarios_per_row) {
                throw std::invalid_argument(
                    "Insufficient American gradient scenario capacity."
                );
            }
            if (sensitivity_count != 0U) {
                validate_device_pointer(primary.stencils, "gradient stencils");
                validate_device_pointer(gradients, "device gradients");
                validate_device_pointer(
                    gradient_errors, "device gradient errors"
                );
                if (primary.stencil_capacity
                    < results * sensitivity_count) {
                    throw std::invalid_argument(
                        "Insufficient American gradient stencil capacity."
                    );
                }
            }
        }

        template<typename Policy, typename Time>
        __device__ __forceinline__ typename Policy::PreparedRow prepare_row(
            std::size_t result_index,
            const Time& time,
            philox::PhiloxKey key,
            std::size_t output_index,
            std::size_t state_offset,
            std::size_t paths_per_price
        ) const {
            const std::size_t scenarios_per_row =
                1U + 2U * sensitivity_count;
            const Scenario* central = primary.scenarios
                + result_index * scenarios_per_row;
            const auto* row_stencils = sensitivity_count == 0U
                ? nullptr
                : primary.stencils + result_index * sensitivity_count;
            return Policy::prepare_gradient_row(
                central,
                row_stencils,
                sensitivity_count,
                key,
                output_result_offset + output_index,
                state_offset,
                paths_per_price,
                time
            );
        }
    };

    struct PreparedRow : Base::PreparedRow {
        const Scenario* scenarios;
        const ::ai_factory::workbench::price_gradients::Stencil* stencils;
        std::size_t sensitivity_count;
        Exercise* exercises;
        InitialDecision* initial_exercise;
        float dt;
        float first_exercise_time;
        float exercise_interval;
        std::uint32_t initial_transition_count;
        std::uint32_t transitions_per_exercise;
    };

    struct StateView : Base::StateView {
        Exercise* exercises;
        InitialDecision* initial_exercises;
    };

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    path_field_descriptors() {
        return {{sizeof(Exercise), alignof(Exercise)}};
    }

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    row_field_descriptors() {
        return {{sizeof(InitialDecision), alignof(InitialDecision)}};
    }

    static std::size_t moment_value_count(const HostInputs& inputs) {
        return longstaff_schwartz::price_gradients::moment_value_count(
            inputs.sensitivity_count
        );
    }

    static StateView make_state_view(
        unsigned char* workspace,
        const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        return {
            Base::make_state_view(workspace, layout),
            longstaff_schwartz::workspace_pointer<Exercise>(
                workspace, layout.path_fields.at(0)
            ),
            longstaff_schwartz::workspace_pointer<InitialDecision>(
                workspace, layout.row_fields.at(0)
            ),
        };
    }

    static longstaff_schwartz::EarlyExerciseRowPlan plan_row(
        const HostInputs& inputs,
        std::size_t result_index,
        std::size_t paths_per_price,
        const TimeConfiguration& time
    ) {
        typename Base::HostInputs base{
            &inputs.product(result_index), 1U, PriceConstruction::Aligned
        };
        return Base::plan_row(base, 0U, paths_per_price, time);
    }

    __device__ __forceinline__ static PreparedRow prepare_gradient_row(
        const Scenario* scenarios,
        const ::ai_factory::workbench::price_gradients::Stencil* stencils,
        std::size_t sensitivity_count,
        philox::PhiloxKey key,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t paths_per_price,
        const TimeConfiguration& time
    ) {
        const Scenario& central = scenarios[0];
        const simulation::MaturityAlignedExerciseCalendar calendar{
            central.product.maturity_days,
            central.product.exercise_interval_days,
        };
        const std::uint32_t first_days =
            simulation::maturity_aligned_first_exercise_days(calendar);
        const float first_time = simulation::day_count_year_fraction(
            first_days, time
        );
        const float interval = simulation::day_count_year_fraction(
            central.product.exercise_interval_days, time
        );
        const typename Base::PreparedRow base = Base::prepare_row(
            central.model,
            central.product,
            key,
            result_index,
            state_offset,
            paths_per_price,
            time
        );
        return {
            base,
            scenarios,
            stencils,
            sensitivity_count,
            nullptr,
            nullptr,
            time.dt,
            first_time,
            interval,
            time.simulation_steps_per_day * first_days,
            time.simulation_steps_per_day
                * central.product.exercise_interval_days,
        };
    }

    __device__ __forceinline__ static void prepare_path_outputs(
        PreparedRow& row,
        const StateView& states,
        std::size_t batch_price,
        std::size_t paths
    ) {
        row.exercises = states.exercises + batch_price * paths;
        row.initial_exercise = states.initial_exercises + batch_price;
        *row.initial_exercise = InitialDecision::continuation;
    }

    __device__ __forceinline__ static float simulate_path(
        const PreparedRow& row,
        std::size_t path,
        std::size_t paths,
        const StateView& states
    ) {
        return simulate_frozen_exercise_path<Schedule, Side, Continuation>(
            row.schedule,
            row.key,
            row.state_offset,
            row.regression_count,
            row.strike,
            row.exercises,
            path,
            paths,
            states
        );
    }

    __device__ __forceinline__ static void record_exercise(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation,
        std::size_t path,
        std::uint32_t backward_level
    ) {
        record_frozen_exercise<Continuation>(
            row.exercises,
            states,
            observation,
            path,
            row.regression_count,
            backward_level
        );
    }

    __device__ __forceinline__ static void record_initial_exercise(
        const PreparedRow& row,
        bool exercise,
        bool valid
    ) {
        *row.initial_exercise = !valid
            ? InitialDecision::invalid
            : exercise ? InitialDecision::exercise
                       : InitialDecision::continuation;
    }

    __device__ __forceinline__ static PreparedSensitivity
    prepare_sensitivity(const PreparedRow& row, std::size_t sensitivity) {
        const Scenario& central = row.scenarios[0];
        const Scenario& first = row.scenarios[1U + 2U * sensitivity];
        const Scenario& second = row.scenarios[2U + 2U * sensitivity];
        return Evaluator::prepare(
            central,
            first,
            second,
            row.stencils[sensitivity],
            row.dt,
            row.first_exercise_time,
            row.exercise_interval,
            row.initial_discount,
            row.exercise_discount
        );
    }

    __device__ __forceinline__ static float path_gradient(
        const PreparedRow& row,
        const PreparedSensitivity& prepared,
        std::size_t path
    ) {
        return Evaluator::path_gradient(
            prepared,
            row.exercises[path],
            row.key,
            path,
            row.initial_transition_count,
            row.transitions_per_exercise
        );
    }

    __device__ __forceinline__ static float initial_gradient(
        const PreparedRow&,
        const PreparedSensitivity& prepared
    ) {
        return Evaluator::initial_gradient(prepared);
    }

    static std::size_t finish_batch(
        const DeviceInputs& inputs,
        const PreparedRow* rows,
        dim3 grid,
        unsigned int threads,
        std::size_t paths,
        std::size_t blocks,
        double* partials,
        const longstaff_schwartz::RegressionDiagnostics* diagnostics,
        const char* name
    ) {
        return longstaff_schwartz::price_gradients::
            finish_frozen_gradient_batch<AmericanOptionPriceGradientPolicy>(
                inputs,
                rows,
                grid,
                threads,
                paths,
                blocks,
                partials,
                diagnostics,
                name
            );
    }
};

}  // namespace ai_factory::workbench::product
