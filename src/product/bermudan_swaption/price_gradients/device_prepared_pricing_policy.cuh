// Bermudan sensitivities from one central LSM policy and row-local GPU tasks.
#pragma once

#include "common/longstaff_schwartz/frozen_exercise_trace.cuh"
#include "common/longstaff_schwartz/price_gradients/device_prepared_frozen_replay_kernels.cuh"
#include "common/longstaff_schwartz/price_gradients/exercise_replay.cuh"
#include "common/longstaff_schwartz/price_gradients/workspace.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/reconstruction.cuh"
#include "common/result_index.cuh"

#include <cuda_runtime.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace ai_factory::workbench::product {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace mcpg = ::ai_factory::workbench::monte_carlo::price_gradients;
namespace sensitivity_preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

template<typename Dynamics>
struct StopAtExerciseObservation {
    std::uint32_t target;

    __device__ __forceinline__ bool on_initial_state(
        const typename Dynamics::State&
    ) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename Dynamics::State&
    ) {
        return observation < target;
    }
};

template<
    typename BasePricingPolicy,
    typename PreparationPolicy,
    typename PrimaryInputsT,
    pg::SensitivityOrders Orders,
    bool TerminalForward,
    typename RegressorT,
    typename ExerciseReplayPolicy
>
struct BermudanSwaptionDevicePreparedSensitivityPolicy
    : BasePricingPolicy {
    static_assert(Orders != pg::SensitivityOrders::none);
    static constexpr std::size_t kNodeCapacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    static constexpr pg::SensitivityOrders request_orders = Orders;
    static constexpr bool kCanExerciseAtInitialTime = false;
    static constexpr bool kTerminalForward = TerminalForward;
    using Base = BasePricingPolicy;
    using Preparation = PreparationPolicy;
    using PrimaryInputs = PrimaryInputsT;
    using Regressor = RegressorT;
    using ExerciseReplay = ExerciseReplayPolicy;
    using RegressionSnapshot = longstaff_schwartz::price_gradients::
        FrozenRegressionSnapshot<Regressor>;
    static constexpr bool kFrozenRegressionPolicy =
        longstaff_schwartz::price_gradients::
            is_frozen_regression_policy_v<ExerciseReplay>;
    using ModelParameters = typename Base::ModelParameters;
    using ProductParameters = typename Base::ProductParameters;
    using Schedule = typename Base::Schedule;
    using Dynamics = typename Base::Dynamics;
    using Analytics = typename Base::Analytics;
    using TimeConfiguration = typename Base::TimeConfiguration;
    using Scenario = typename Preparation::Scenario;
    using Exercise = longstaff_schwartz::FrozenExerciseIndex;
    using InitialDecision = longstaff_schwartz::InitialExerciseDecision;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;

    struct HostInputs {
        const ProductParameters* products;
        std::size_t product_count;
        PriceConstruction construction;
        std::size_t result_offset;
        std::size_t total_result_count;
        std::size_t sensitivity_count;

        void validate(std::size_t results) const {
            if (products == nullptr || product_count == 0U || results == 0U
                || result_offset >= total_result_count
                || results > total_result_count - result_offset) {
                throw std::invalid_argument(
                    "Malformed Bermudan sensitivity host plan."
                );
            }
        }

        const ProductParameters& product(std::size_t local_row) const {
            const std::size_t row = result_offset + local_row;
            const std::size_t product_index = is_cartesian(construction)
                ? row % product_count
                : row;
            return products[product_index];
        }
    };

    struct DeviceInputs {
        PrimaryInputs primary;
        mcpg::DevicePreparedStencilOutputs<kNodeCapacity> stencil_outputs;
        PriceConstruction construction;
        pg::TimeConfiguration time;
        std::size_t model_count;
        std::size_t curve_count;
        std::size_t product_count;
        std::size_t total_result_count;
        std::size_t input_result_offset;
        std::size_t sensitivity_count;
        std::uint32_t maximum_exercise_count;
        pg::SensitivityOutputs outputs;

        void validate_inputs_and_tasks(std::size_t results) const {
            validate_device_pointer(primary.models, "sensitivity models");
            validate_device_pointer(primary.products, "sensitivity products");
            bool insufficient = primary.model_capacity < model_count
                || primary.product_capacity < product_count
                || input_result_offset >= total_result_count
                || results > total_result_count - input_result_offset;
            if constexpr (requires { primary.curves; }) {
                validate_device_pointer(primary.curves, "sensitivity curves");
                insufficient = insufficient
                    || primary.curve_capacity < curve_count;
            }
            if (insufficient) {
                throw std::invalid_argument(
                    "Insufficient Bermudan sensitivity input capacity."
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
                    < total_result_count * sensitivity_count
                || maximum_exercise_count < 2U) {
                throw std::invalid_argument(
                    "Insufficient Bermudan sensitivity task capacity."
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
                    "Insufficient Bermudan sensitivity output capacity."
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
            const mcpg::DevicePreparedPlan plan{
                time,
                construction,
                model_count,
                product_count,
                total_result_count,
                sensitivity_count,
                curve_count,
            };
            Scenario central{};
            const bool valid = primary.make_central(
                result_index, plan, central
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

    struct PreparedRow : Base::PreparedRow {
        Scenario central;
        const pg::SensitivitySpec<typename Preparation::Parameter>*
            sensitivities;
        Stencil* represented_stencils;
        sensitivity_preparation::Error* preparation_error;
        std::size_t sensitivity_count;
        Exercise* exercises;
        RegressionSnapshot* frozen_regressions;
        pg::TimeConfiguration time;
        TimeConfiguration simulation_time;
    };

    struct PreparedSensitivity {
        typename Base::PreparedRow nodes[kNodeCapacity];
        Stencil stencil;
        std::uint8_t node_count = 0U;
        bool valid = false;
    };

    struct StateView : Base::StateView {
        Exercise* exercises;
        RegressionSnapshot* frozen_regressions;
    };

    template<typename HostPlan>
    static HostInputs make_host_inputs(
        const HostPlan& host,
        std::size_t result_offset
    ) {
        return {
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
        std::size_t curve_count = 0U;
        if constexpr (requires { host.curves.size(); }) {
            curve_count = host.curves.size();
        }
        return {
            device,
            stencil_outputs,
            host.construction,
            host.time,
            host.models.size(),
            curve_count,
            host.products.size(),
            host.result_count,
            result_offset,
            host.sensitivity_count(),
            host.maximum_exercise_count,
            outputs,
        };
    }

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    observation_field_descriptors() {
        std::vector<longstaff_schwartz::StateFieldDescriptor> descriptors;
        if constexpr (requires { Base::observation_field_descriptors(); }) {
            descriptors = Base::observation_field_descriptors();
        }
        if constexpr (kFrozenRegressionPolicy) {
            descriptors = longstaff_schwartz::price_gradients::
                with_frozen_regression_field<Regressor>(
                    std::move(descriptors)
                );
        }
        return descriptors;
    }

    static std::vector<longstaff_schwartz::StateFieldDescriptor>
    path_field_descriptors() {
        if constexpr (kFrozenRegressionPolicy) return {};
        return {{sizeof(Exercise), alignof(Exercise)}};
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
                    workspace, layout.observation_fields.back()
                );
        } else {
            exercises = longstaff_schwartz::workspace_pointer<Exercise>(
                workspace, layout.path_fields.at(0)
            );
        }
        return {
            Base::make_state_view(workspace, layout),
            exercises,
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
            &inputs.product(result_index),
            1U,
            PriceConstruction::Aligned,
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
        if constexpr (requires { scenario.curve; }) {
            return Base::prepare_row(
                scenario.model,
                scenario.curve,
                scenario.product,
                key,
                result_index,
                state_offset,
                paths_per_price,
                time
            );
        } else {
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
        const TimeConfiguration& simulation_time
    ) {
        const typename Base::PreparedRow base = prepare_node_row(
            central,
            key,
            result_index,
            state_offset,
            paths_per_price,
            simulation_time
        );
        return {
            base,
            central,
            sensitivities,
            stencil_outputs.stencils,
            stencil_outputs.error,
            sensitivity_count,
            nullptr,
            nullptr,
            sensitivity_time,
            simulation_time,
        };
    }

    __device__ __forceinline__ static void prepare_observations(
        PreparedRow& row,
        const StateView& states
    ) {
        if constexpr (requires {
            Base::prepare_observations(row, states);
        }) {
            Base::prepare_observations(row, states);
        }
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
    }

    __device__ __forceinline__ static float simulate_path(
        const PreparedRow& row,
        std::size_t path,
        std::size_t paths,
        const StateView& states
    ) {
        if constexpr (!kFrozenRegressionPolicy) {
            row.exercises[path].observation = row.regression_count;
        }
        return Base::simulate_path(row, path, paths, states);
    }

    __device__ __forceinline__ static void record_exercise(
        const PreparedRow& row,
        const StateView&,
        std::size_t observation,
        std::size_t path,
        std::uint32_t
    ) {
        if constexpr (!kFrozenRegressionPolicy) {
            row.exercises[path].observation =
                Base::exercise_from_state_index(row, observation);
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

    __device__ __forceinline__ static InitialDecision
    initial_exercise_decision(const PreparedRow&) {
        return InitialDecision::continuation;
    }

    static std::size_t sensitivity_workspace_bytes(
        const DeviceInputs& inputs
    ) {
        if constexpr (TerminalForward) {
            return kNodeCapacity
                * static_cast<std::size_t>(inputs.maximum_exercise_count)
                * sizeof(typename Schedule::Observation);
        }
        return 0U;
    }

    __device__ __forceinline__ static PreparedSensitivity
    prepare_sensitivity(
        const PreparedRow& row,
        std::size_t sensitivity,
        bool write_stencil,
        unsigned char* workspace,
        std::size_t workspace_bytes
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

        PreparedSensitivity prepared{};
        prepared.stencil = task.stencil;
        prepared.node_count = static_cast<std::uint8_t>(
            pg::active_node_count(task.stencil)
        );
        if constexpr (TerminalForward) {
            const std::size_t required =
                static_cast<std::size_t>(prepared.node_count)
                * row.product.exercise_count
                * sizeof(typename Schedule::Observation);
            if (workspace == nullptr || workspace_bytes < required) {
                sensitivity_preparation::record_error(
                    row.preparation_error,
                    sensitivity_preparation::invalid_central,
                    row.result_index,
                    sensitivity
                );
                return {};
            }
        }
        #pragma unroll
        for (std::size_t node = 0U; node < kNodeCapacity; ++node) {
            if (node >= prepared.node_count) continue;
            prepared.nodes[node] = prepare_node_row(
                task.nodes[node],
                row.key,
                row.result_index,
                row.state_offset,
                row.paths_per_price,
                row.simulation_time
            );
            if constexpr (TerminalForward) {
                auto* observations =
                    reinterpret_cast<typename Schedule::Observation*>(
                        workspace
                    ) + node * row.product.exercise_count;
                Base::prepare_observation_table(
                    prepared.nodes[node], observations
                );
            }
        }
        if (write_stencil) {
            row.represented_stencils[
                row.result_index * row.sensitivity_count + sensitivity
            ] = task.stencil;
        }
        prepared.valid = true;
        return prepared;
    }

    __device__ __forceinline__ static PreparedSensitivity
    prepare_sensitivity(
        const PreparedRow& row,
        std::size_t sensitivity,
        bool write_stencil
    ) {
        return prepare_sensitivity(
            row, sensitivity, write_stencil, nullptr, 0U
        );
    }

    __device__ __forceinline__ static float replay_node_value(
        const PreparedRow& row,
        const typename Base::PreparedRow& node,
        std::size_t path
    ) {
        std::uint32_t exercise = row.regression_count;
        typename Dynamics::State state{};
        if constexpr (kFrozenRegressionPolicy) {
            longstaff_schwartz::price_gradients::
                FrozenRegressionExerciseHandler<Base, Regressor> handler{
                    row,
                    node,
                    row.frozen_regressions,
                    row.regression_count,
                };
            state = Schedule::simulate(
                node.schedule, row.key, path, handler
            );
            exercise = handler.exercise;
        } else {
            exercise = row.exercises[path].observation;
            StopAtExerciseObservation<Dynamics> handler{exercise};
            state = Schedule::simulate(
                node.schedule, row.key, path, handler
            );
        }

        if constexpr (TerminalForward) {
            return Base::normalized_payoff(node, state, exercise);
        } else {
            const float payoff = Base::immediate_value_at(
                node,
                Analytics::factor_state(state),
                exercise
            );
            const float log_discount = Analytics::log_discount_factor(
                node.analytics,
                state.state_integral,
                Base::exercise_time(node, exercise)
            );
            return expf(log_discount) * payoff;
        }
    }

    __device__ __forceinline__ static pg::SensitivityResult path_sensitivity(
        const PreparedRow& row,
        const PreparedSensitivity& prepared,
        std::size_t path
    ) {
        pg::SensitivityValues<kNodeCapacity> values{};
        #pragma unroll
        for (std::size_t node = 0U; node < kNodeCapacity; ++node) {
            if (node < prepared.node_count) {
                values[node] = replay_node_value(
                    row, prepared.nodes[node], path
                );
            }
        }
        return pg::reconstruct_sensitivity<Orders>(
            prepared.stencil, values
        );
    }

    __device__ __forceinline__ static pg::SensitivityResult
    initial_sensitivity(const PreparedRow&, const PreparedSensitivity&) {
        return {};
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
                BermudanSwaptionDevicePreparedSensitivityPolicy
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
