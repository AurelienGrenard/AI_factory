// American sensitivities with one central LSM solve and row-local GPU tasks.
#pragma once

#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/longstaff_schwartz/price_gradients/execution_plan.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_frozen_replay_kernels.cuh"
#include "common/longstaff_schwartz/price_gradients/exercise_replay.cuh"
#include "common/longstaff_schwartz/price_gradients/workspace.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include "product/american_option/frozen_exercise_value.cuh"
#include "product/american_option/price_gradients/device_prepared_frozen_exercise_evaluator.cuh"
#include "product/american_option/pricing_policy.cuh"

#include <concepts>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::product {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg =
    ::ai_factory::workbench::monte_carlo::price_gradients;
namespace sensitivity_preparation =
    ::ai_factory::workbench::equity::price_gradients::device_preparation;

template<
    typename SchedulePolicy,
    OptionSide Side,
    typename Continuation,
    typename ReplayPolicy,
    typename PreparationPolicy,
    ::ai_factory::workbench::price_gradients::SensitivityOrders Orders,
    typename RegressorT,
    typename ExerciseReplayPolicy
>
struct AmericanOptionDevicePreparedSensitivityPolicy
    : AmericanOptionPricingPolicy<SchedulePolicy, Side, Continuation> {
    static_assert(Orders != pg::SensitivityOrders::none);
    static constexpr std::size_t kNodeCapacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    static constexpr pg::SensitivityOrders request_orders = Orders;
    static constexpr bool kCanExerciseAtInitialTime = true;
    using Base = AmericanOptionPricingPolicy<
        SchedulePolicy, Side, Continuation
    >;
    using Schedule = SchedulePolicy;
    using Preparation = PreparationPolicy;
    using Replay = ReplayPolicy;
    using Regressor = RegressorT;
    using ExerciseReplay = ExerciseReplayPolicy;
    using Dynamics = typename Base::Dynamics;
    using RegressionSnapshot = longstaff_schwartz::price_gradients::
        FrozenRegressionSnapshot<Regressor>;
    static constexpr bool kFrozenRegressionPolicy =
        longstaff_schwartz::price_gradients::
            is_frozen_regression_policy_v<ExerciseReplay>;
    static constexpr OptionSide kSide = Side;
    using ModelParameters = typename Base::ModelParameters;
    using ProductParameters = typename Base::ProductParameters;
    using TimeConfiguration = typename Base::TimeConfiguration;
    using Scenario = typename Preparation::Scenario;
    using Evaluator = american_option::price_gradients::
        DevicePreparedFrozenExerciseEvaluator<Side, Orders, ReplayPolicy>;
    using Exercise = longstaff_schwartz::FrozenExerciseTrace;
    using InitialDecision = longstaff_schwartz::InitialExerciseDecision;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;

    struct HostInputs {
        const ModelParameters* models;
        std::size_t model_count;
        const ProductParameters* products;
        std::size_t product_count;
        PriceConstruction construction;
        std::size_t result_offset;
        std::size_t total_result_count;
        std::size_t sensitivity_count;

        void validate(std::size_t results) const {
            if (models == nullptr || products == nullptr
                || model_count == 0U || product_count == 0U
                || results == 0U
                || result_offset >= total_result_count
                || results > total_result_count - result_offset) {
                throw std::invalid_argument(
                    "Malformed American sensitivity host plan."
                );
            }
        }

        const ProductParameters& product(std::size_t local_row) const {
            const auto indices = pg::price_row_indices(
                result_offset + local_row, construction, product_count
            );
            return products[indices.product];
        }
    };

    struct DeviceInputs {
        mcpg::DevicePreparedInputs<Preparation> primary;
        mcpg::DevicePreparedStencilOutputs<kNodeCapacity> stencil_outputs;
        PriceConstruction construction;
        pg::TimeConfiguration time;
        std::size_t model_count;
        std::size_t product_count;
        std::size_t total_result_count;
        std::size_t input_result_offset;
        std::size_t sensitivity_count;
        pg::SensitivityOutputs outputs;

        std::size_t maximum_batch_size(std::size_t maximum_grid_y) const {
            return longstaff_schwartz::price_gradients::maximum_batch_size(
                sensitivity_count, maximum_grid_y
            );
        }

        void validate_batch_grid(
            std::size_t batch_size, std::size_t maximum_grid_y
        ) const {
            longstaff_schwartz::price_gradients::validate_task_grid(
                batch_size, sensitivity_count, maximum_grid_y
            );
        }

        void validate_inputs_and_tasks(std::size_t results) const {
            validate_device_pointer(primary.models, "sensitivity models");
            validate_device_pointer(primary.products, "sensitivity products");
            if (primary.model_capacity < model_count
                || primary.product_capacity < product_count
                || input_result_offset >= total_result_count
                || results > total_result_count - input_result_offset) {
                throw std::invalid_argument(
                    "Insufficient American sensitivity input capacity."
                );
            }
            if (sensitivity_count == 0U) return;
            validate_device_pointer(
                primary.sensitivities, "sensitivity specifications"
            );
            validate_device_pointer(
                stencil_outputs.stencils, "represented stencils"
            );
            validate_device_pointer(
                stencil_outputs.error, "sensitivity preparation status"
            );
            if (primary.sensitivity_capacity < sensitivity_count
                || stencil_outputs.capacity
                    < total_result_count * sensitivity_count) {
                throw std::invalid_argument(
                    "Insufficient American sensitivity task capacity."
                );
            }
        }

        void validate(std::size_t results) const {
            validate_inputs_and_tasks(results);
            if (sensitivity_count == 0U) return;
            if constexpr (pg::requests_first_v<Orders>) {
                validate_device_pointer(outputs.gradients, "gradients");
                validate_device_pointer(
                    outputs.gradient_standard_errors,
                    "gradient standard errors"
                );
            }
            if constexpr (pg::requests_second_v<Orders>) {
                validate_device_pointer(
                    outputs.diagonal_hessians, "diagonal Hessians"
                );
                validate_device_pointer(
                    outputs.diagonal_hessian_standard_errors,
                    "diagonal Hessian standard errors"
                );
            }
            if (outputs.sensitivity_capacity
                < total_result_count * sensitivity_count) {
                throw std::invalid_argument(
                    "Insufficient American sensitivity output capacity."
                );
            }
        }

        template<typename Policy, typename Time>
        __device__ __forceinline__ typename Policy::PreparedRow prepare_row(
            std::size_t local_row,
            const Time& time_configuration,
            philox::PhiloxKey key,
            std::size_t,
            std::size_t state_offset,
            std::size_t paths_per_price
        ) const {
            const std::size_t result_index =
                input_result_offset + local_row;
            const auto indices = pg::price_row_indices(
                result_index, construction, product_count
            );
            Scenario central{};
            const bool valid = Preparation::make_central(
                primary.models[indices.model],
                primary.products[indices.product],
                time,
                central
            );
            if (!valid) {
                sensitivity_preparation::record_error(
                    stencil_outputs.error,
                    sensitivity_preparation::invalid_central,
                    result_index,
                    0U
                );
            }
            return Policy::prepare_sensitivity_row(
                central,
                primary.sensitivities,
                sensitivity_count,
                stencil_outputs,
                key,
                result_index,
                state_offset,
                paths_per_price,
                time,
                time_configuration
            );
        }
    };

    template<typename HostPlan>
    static HostInputs make_host_inputs(
        const HostPlan& host,
        std::size_t result_offset
    ) {
        return {
            host.models.data(),
            host.models.size(),
            host.products.data(),
            host.products.size(),
            host.construction,
            result_offset,
            host.result_count,
            host.sensitivity_count(),
        };
    }

    template<typename HostPlan>
    static DeviceInputs make_device_inputs(
        const HostPlan& host,
        typename HostPlan::DeviceInputs device,
        mcpg::DevicePreparedStencilOutputs<kNodeCapacity> stencil_outputs,
        std::size_t result_offset,
        pg::SensitivityOutputs outputs
    ) {
        return {
            device,
            stencil_outputs,
            host.construction,
            host.time,
            host.models.size(),
            host.products.size(),
            host.result_count,
            result_offset,
            host.sensitivity_count(),
            outputs,
        };
    }

    struct PreparedRow : Base::PreparedRow {
        Scenario central;
        const pg::SensitivitySpec<typename Preparation::Parameter>*
            sensitivities;
        Stencil* represented_stencils;
        sensitivity_preparation::Error* preparation_error;
        std::size_t sensitivity_count;
        std::size_t paths_per_price;
        Exercise* exercises;
        InitialDecision* initial_exercise;
        double* initial_continuation;
        RegressionSnapshot* frozen_regressions;
        pg::TimeConfiguration time;
        TimeConfiguration simulation_time;
        float first_exercise_time;
        float exercise_interval;
        std::uint32_t initial_transition_count;
        std::uint32_t transitions_per_exercise;
    };

    struct FrozenExercisePreparedSensitivity {
        typename Evaluator::template Prepared<Scenario> value;
        bool valid = false;
    };

    struct FrozenRegressionPreparedSensitivity {
        typename Base::PreparedRow nodes[kNodeCapacity];
        Stencil stencil;
        std::uint8_t node_count = 0U;
        bool valid = false;
    };

    using PreparedSensitivity = std::conditional_t<
        kFrozenRegressionPolicy,
        FrozenRegressionPreparedSensitivity,
        FrozenExercisePreparedSensitivity
    >;

    struct StateView : Base::StateView {
        Exercise* exercises;
        InitialDecision* initial_exercises;
        double* initial_continuations;
        RegressionSnapshot* frozen_regressions;
    };

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    observation_field_descriptors() {
        if constexpr (kFrozenRegressionPolicy) {
            return longstaff_schwartz::price_gradients::
                with_frozen_regression_field<Regressor>();
        }
        return {};
    }

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    path_field_descriptors() {
        if constexpr (kFrozenRegressionPolicy) return {};
        return {{sizeof(Exercise), alignof(Exercise)}};
    }

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    row_field_descriptors() {
        std::vector<longstaff_schwartz::StateFieldDescriptor> descriptors{
            {sizeof(InitialDecision), alignof(InitialDecision)}
        };
        if constexpr (kFrozenRegressionPolicy) {
            descriptors.push_back({sizeof(double), alignof(double)});
        }
        return descriptors;
    }

    static std::size_t moment_value_count(const HostInputs& inputs) {
        return longstaff_schwartz::price_gradients::moment_value_count(
            inputs.sensitivity_count,
            longstaff_schwartz::price_gradients::
                kSensitivityMomentCount<Orders>
        );
    }

    static StateView make_state_view(
        unsigned char* workspace,
        const longstaff_schwartz::WorkspaceLayout& layout
    ) {
        Exercise* exercises = nullptr;
        RegressionSnapshot* frozen_regressions = nullptr;
        if constexpr (kFrozenRegressionPolicy) {
            frozen_regressions =
                longstaff_schwartz::workspace_pointer<RegressionSnapshot>(
                    workspace, layout.observation_fields.at(0)
                );
        } else {
            exercises = longstaff_schwartz::workspace_pointer<Exercise>(
                workspace, layout.path_fields.at(0)
            );
        }
        return {
            Base::make_state_view(workspace, layout),
            exercises,
            longstaff_schwartz::workspace_pointer<InitialDecision>(
                workspace, layout.row_fields.at(0)
            ),
            kFrozenRegressionPolicy
                ? longstaff_schwartz::workspace_pointer<double>(
                      workspace, layout.row_fields.at(1)
                  )
                : nullptr,
            frozen_regressions,
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

    __device__ __forceinline__ static typename Base::PreparedRow
    prepare_node_row(
        const Scenario& scenario,
        philox::PhiloxKey key,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t paths_per_price,
        const TimeConfiguration& time
    ) {
        return Base::prepare_row(
            scenario.model,
            scenario.product,
            key,
            result_index,
            state_offset,
            paths_per_price,
            time
        );
    }

    __device__ __forceinline__ static PreparedRow prepare_sensitivity_row(
        const Scenario& central,
        const pg::SensitivitySpec<typename Preparation::Parameter>*
            sensitivities,
        std::size_t sensitivity_count,
        mcpg::DevicePreparedStencilOutputs<kNodeCapacity> stencil_outputs,
        philox::PhiloxKey key,
        std::size_t result_index,
        std::size_t state_offset,
        std::size_t paths_per_price,
        const pg::TimeConfiguration& sensitivity_time,
        const TimeConfiguration& time
    ) {
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
        std::uint32_t initial_transition_count = 0U;
        std::uint32_t transitions_per_exercise = 0U;
        if constexpr (std::same_as<
                          TimeConfiguration,
                          simulation::FixedStepTimeConfiguration>) {
            initial_transition_count =
                time.simulation_steps_per_day * first_days;
            transitions_per_exercise =
                time.simulation_steps_per_day
                * central.product.exercise_interval_days;
        }
        return {
            base,
            central,
            sensitivities,
            stencil_outputs.stencils,
            stencil_outputs.error,
            sensitivity_count,
            paths_per_price,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            sensitivity_time,
            time,
            first_time,
            interval,
            initial_transition_count,
            transitions_per_exercise,
        };
    }

    __device__ __forceinline__ static void prepare_observations(
        PreparedRow& row,
        const StateView& states
    ) {
        if constexpr (kFrozenRegressionPolicy) {
            row.frozen_regressions = states.frozen_regressions
                + row.state_offset / row.paths_per_price;
        }
    }

    __device__ __forceinline__ static void prepare_path_outputs(
        PreparedRow& row,
        const StateView& states,
        std::size_t batch_price,
        std::size_t paths
    ) {
        if constexpr (!kFrozenRegressionPolicy) {
            row.exercises = states.exercises + batch_price * paths;
        }
        row.initial_exercise = states.initial_exercises + batch_price;
        *row.initial_exercise = InitialDecision::continuation;
        if constexpr (kFrozenRegressionPolicy) {
            row.initial_continuation =
                states.initial_continuations + batch_price;
            *row.initial_continuation = nan("");
        }
    }

    __device__ __forceinline__ static float simulate_path(
        const PreparedRow& row,
        std::size_t path,
        std::size_t paths,
        const StateView& states
    ) {
        if constexpr (kFrozenRegressionPolicy) {
            return Base::simulate_path(row, path, paths, states);
        } else {
            return simulate_frozen_exercise_path<
                Schedule, Side, Continuation
            >(
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
    }

    __device__ __forceinline__ static void record_exercise(
        const PreparedRow& row,
        const StateView& states,
        std::size_t observation,
        std::size_t path,
        std::uint32_t backward_level
    ) {
        if constexpr (!kFrozenRegressionPolicy) {
            record_frozen_exercise<Continuation>(
                row.exercises,
                states,
                observation,
                path,
                row.regression_count,
                backward_level
            );
        }
    }

    __device__ __forceinline__ static void record_regression(
        const PreparedRow& row,
        const StateView&,
        std::uint32_t backward_level,
        const double* coefficients,
        longstaff_schwartz::RegressionStatus status
    ) requires(kFrozenRegressionPolicy) {
        longstaff_schwartz::price_gradients::capture_frozen_regression<
            Regressor
        >(
            row.frozen_regressions,
            row.regression_count,
            backward_level,
            coefficients,
            status
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

    __device__ __forceinline__ static void record_initial_continuation(
        const PreparedRow& row,
        double continuation,
        bool valid
    ) requires(kFrozenRegressionPolicy) {
        *row.initial_continuation = valid ? continuation : nan("");
    }

    __device__ __forceinline__ static PreparedSensitivity
    prepare_sensitivity(
        const PreparedRow& row,
        std::size_t sensitivity,
        bool write_stencil
    ) {
        pg::SensitivityTask<Scenario, kNodeCapacity> task{};
        int error = sensitivity_preparation::valid;
        const bool valid = sensitivity_preparation::build_sensitivity_task<
            Orders,
            Preparation
        >(
            row.central,
            row.sensitivities[sensitivity],
            row.time,
            task,
            error
        );
        if (!valid) {
            sensitivity_preparation::record_error(
                row.preparation_error,
                error,
                row.result_index,
                sensitivity
            );
            return {};
        }
        if (write_stencil) {
            row.represented_stencils[
                row.result_index * row.sensitivity_count + sensitivity
            ] = task.stencil;
        }

        if constexpr (kFrozenRegressionPolicy) {
            PreparedSensitivity prepared{};
            prepared.stencil = task.stencil;
            prepared.node_count = static_cast<std::uint8_t>(
                pg::active_node_count(task.stencil)
            );
            #pragma unroll
            for (std::size_t node = 0U;
                 node < kNodeCapacity;
                 ++node) {
                if (node < prepared.node_count) {
                    prepared.nodes[node] = prepare_node_row(
                        task.nodes[node],
                        row.key,
                        row.result_index,
                        row.state_offset,
                        row.paths_per_price,
                        row.simulation_time
                    );
                }
            }
            prepared.valid = true;
            return prepared;
        } else {
            return {
                Evaluator::prepare(
                    task,
                    row.time.dt,
                    row.first_exercise_time,
                    row.exercise_interval,
                    row.initial_discount,
                    row.exercise_discount
                ),
                true,
            };
        }
    }

    __device__ __forceinline__ static float replay_node_value(
        const PreparedRow& row,
        const typename Base::PreparedRow& node,
        std::size_t path
    ) {
        const float initial_exercise = Base::initial_exercise_value(node);
        if (longstaff_schwartz::exercise_is_preferred(
                initial_exercise, *row.initial_continuation
            )) {
            return initial_exercise;
        }

        longstaff_schwartz::price_gradients::
            FrozenRegressionExerciseHandler<Base, Regressor> handler{
                row,
                node,
                row.frozen_regressions,
                row.regression_count,
            };
        const typename Dynamics::State state = Schedule::simulate(
            node.schedule, row.key, path, handler
        );
        float value = payoff::vanilla_option_payoff<Side>(
            Dynamics::spot(state), node.strike
        );
        for (std::uint32_t date = 0U;
             date < handler.exercise;
             ++date) {
            value = node.exercise_discount * value;
        }
        return node.initial_discount * value;
    }

    __device__ __forceinline__ static pg::SensitivityResult path_sensitivity(
        const PreparedRow& row,
        const PreparedSensitivity& prepared,
        std::size_t path
    ) {
        if constexpr (kFrozenRegressionPolicy) {
            pg::SensitivityValues<kNodeCapacity> values{};
            #pragma unroll
            for (std::size_t node = 0U;
                 node < kNodeCapacity;
                 ++node) {
                if (node < prepared.node_count) {
                    values[node] = replay_node_value(
                        row, prepared.nodes[node], path
                    );
                }
            }
            return pg::reconstruct_sensitivity<Orders>(
                prepared.stencil, values
            );
        } else {
            return Evaluator::evaluate_path(
                prepared.value,
                row.exercises[path],
                row.key,
                path,
                row.initial_transition_count,
                row.transitions_per_exercise
            );
        }
    }

    __device__ __forceinline__ static pg::SensitivityResult
    initial_sensitivity(
        const PreparedRow&,
        const PreparedSensitivity& prepared
    ) {
        if constexpr (kFrozenRegressionPolicy) {
            pg::SensitivityValues<kNodeCapacity> values{};
            #pragma unroll
            for (std::size_t node = 0U;
                 node < kNodeCapacity;
                 ++node) {
                if (node < prepared.node_count) {
                    values[node] = Base::initial_exercise_value(
                        prepared.nodes[node]
                    );
                }
            }
            return pg::reconstruct_sensitivity<Orders>(
                prepared.stencil, values
            );
        } else {
            return Evaluator::evaluate_initial(prepared.value);
        }
    }

    static std::size_t sensitivity_workspace_bytes(const DeviceInputs&) {
        return 0U;
    }

    __device__ __forceinline__ static InitialDecision
    initial_exercise_decision(const PreparedRow& row) {
        return *row.initial_exercise;
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
            finish_device_prepared_frozen_replay_sensitivity_batch<
                Orders,
                AmericanOptionDevicePreparedSensitivityPolicy
            >(
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
