// One block per (row, sensitivity), with finite-difference nodes built on device.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/philox.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/monte_carlo/price_gradients/device_prepared_stencil_kernel.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::equity::price_gradients::device_preparation;

namespace detail {

template<typename Dynamics, typename ProductPolicy, typename Preparation>
struct FirstTerminalSensitivityPolicy {
    using Scenario = typename Preparation::Scenario;
    using Stencil = pg::SensitivityStencil<3U>;
    using Nodes = pg::SensitivityNodes<Scenario, 3U>;

    struct PreparedRow {
        typename Dynamics::Prepared dynamics[3U];
        typename ProductPolicy::PreparedProduct products[3U];
        const Nodes* nodes;
        const Stencil* stencil;
        std::uint32_t maximum_steps;

        __device__ __forceinline__ const Scenario& input(
            unsigned int node
        ) const {
            return (*nodes)[node];
        }
    };

    struct Observation {
        struct State {
            float unscaled_spot;
            float scale;
        };

        __device__ __forceinline__ static float spot(const State& state) {
            return state.unscaled_spot * state.scale;
        }
    };

    template<bool IncludeCentral>
    __device__ __forceinline__ static float terminal_spot(
        const PreparedRow& row,
        const typename Dynamics::State* states,
        unsigned int node
    ) {
        if constexpr (IncludeCentral) {
            return Dynamics::spot(
                row.input(node).reuse_central ? states[0U] : states[node]
            );
        } else {
            return Dynamics::spot(states[node - 1U]);
        }
    }

    template<bool IncludeCentral>
    __device__ __forceinline__ static float finalize(
        const PreparedRow& row,
        const typename Dynamics::State* states,
        unsigned int node
    ) {
        const auto handler = ProductPolicy::make_handler(row.products[node]);
        const typename Observation::State terminal{
            terminal_spot<IncludeCentral>(row, states, node),
            row.input(node).spot_scale,
        };
        return ProductPolicy::template finalize<Observation>(
            row.products[node], terminal, handler
        );
    }

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time,
        bool include_central
    ) {
        PreparedRow row{};
        row.nodes = nodes;
        row.stencil = stencil;
        #pragma unroll
        for (unsigned int node = 0U; node < 3U; ++node) {
            const auto& input = row.input(node);
            auto model = input.model;
            model.spot = input.simulation_spot;
            const float horizon = Dynamics::kExactTerminal
                ? input.maturity_years
                : time.dt;
            if (node == 0U && !include_central) {
                if constexpr (Dynamics::kDrawRequiresCentralPrepared) {
                    row.dynamics[0U] = Dynamics::prepare(model, horizon);
                }
                continue;
            }
            row.dynamics[node] = Dynamics::prepare(model, horizon);
            row.products[node] = ProductPolicy::prepare_product(
                input.model,
                input.product,
                {
                    static_cast<float>(time.simulation_steps_per_day) * time.dt,
                    input.maturity_years,
                }
            );
            row.maximum_steps = max(row.maximum_steps, input.step_count);
        }
        return row;
    }

    struct PathResult {
        float price;
        float sensitivity;
    };

    template<bool IncludeCentral, bool IncludePayoff>
    __device__ __forceinline__ static PathResult evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        static_assert(IncludeCentral || !IncludePayoff);
        constexpr unsigned int first_node = IncludeCentral ? 0U : 1U;
        typename Dynamics::State states[3U - first_node];
        #pragma unroll
        for (unsigned int node = first_node; node < 3U; ++node) {
            states[node - first_node] = Dynamics::initial(row.dynamics[node]);
        }
        typename Dynamics::RandomContext random(key, path);
        simulate_coupled_terminal_nodes<3U, Dynamics>(
            random,
            row.dynamics,
            3U,
            row.maximum_steps,
            [&](unsigned int node) {
                if (node == 0U) return IncludeCentral;
                return !row.input(node).reuse_central;
            },
            [&](unsigned int node) { return row.input(node).step_count; },
            [&](unsigned int node) {
                return row.input(node).normal_weights;
            },
            [&](unsigned int node) -> typename Dynamics::State& {
                return states[node - first_node];
            }
        );
        PathResult result{};
        if constexpr (IncludePayoff) {
            result.price = finalize<IncludeCentral>(row, states, 0U);
        }
        if (row.stencil->kind == pg::StencilKind::centered) {
            result.sensitivity =
                ProductPolicy::template centered_difference<Observation>(
                    row.products[1U],
                    row.products[2U],
                    {
                        terminal_spot<IncludeCentral>(row, states, 1U),
                        row.input(1U).spot_scale,
                    },
                    {
                        terminal_spot<IncludeCentral>(row, states, 2U),
                        row.input(2U).spot_scale,
                    },
                    row.stencil->represented_width
                );
        } else if constexpr (IncludePayoff) {
            pg::SensitivityValues<3U> values{};
            values[0U] = result.price;
            values[1U] = finalize<IncludeCentral>(row, states, 1U);
            values[2U] = finalize<IncludeCentral>(row, states, 2U);
            result.sensitivity = pg::reconstruct_sensitivity<
                pg::SensitivityOrders::first
            >(*row.stencil, values).first;
        }
        return result;
    }
};

template<
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation>
__device__ __noinline__ int prepare_first_row(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    std::size_t row,
    std::size_t sensitivity_index,
    typename FirstTerminalSensitivityPolicy<
        Dynamics, ProductPolicy, Preparation
    >::Nodes* nodes,
    typename FirstTerminalSensitivityPolicy<
        Dynamics, ProductPolicy, Preparation
    >::Stencil* stencil,
    typename FirstTerminalSensitivityPolicy<
        Dynamics, ProductPolicy, Preparation
    >::PreparedRow* prepared,
    pg::CentralRequirement* central_requirement,
    bool computes_central_price,
    DevicePreparedStencilOutputs<3U> stencil_outputs
) {
    using Policy = FirstTerminalSensitivityPolicy<
        Dynamics, ProductPolicy, Preparation
    >;
    const auto indices = pg::price_row_indices(
        row, plan.construction, plan.product_count
    );
    int preparation_error = preparation::valid;
    typename Preparation::Scenario central{};
    int row_valid = Preparation::make_central(
        inputs.models[indices.model],
        inputs.products[indices.product],
        plan.time,
        central
    );
    pg::SensitivityTask<typename Preparation::Scenario, 3U> task{};
    if (row_valid) {
        row_valid = preparation::build_sensitivity_task<
            pg::SensitivityOrders::first,
            Preparation
        >(
            central,
            inputs.sensitivities[sensitivity_index],
            plan.time,
            task,
            preparation_error
        );
    } else {
        preparation_error = preparation::invalid_central;
    }
    if (row_valid) {
        *nodes = task.nodes;
        *stencil = task.stencil;
        *central_requirement = computes_central_price
            ? pg::CentralRequirement::payoff
            : task.central_requirement;
        *prepared = Policy::prepare(
            nodes,
            stencil,
            plan.time,
            *central_requirement != pg::CentralRequirement::none
        );
        stencil_outputs.stencils[
            row * plan.sensitivity_count + sensitivity_index
        ] = *stencil;
    } else {
        preparation::record_error(
            stencil_outputs.error,
            preparation_error,
            row,
            sensitivity_index
        );
    }
    return row_valid;
}

template<
    typename Policy,
    bool ComputesCentralPrice,
    bool IncludeCentral,
    bool IncludePayoff>
__device__ __forceinline__ void evaluate_first_and_reduce(
    const typename Policy::PreparedRow& prepared,
    philox::PhiloxKey key,
    const pg::LaunchConfiguration& launch,
    pg::Outputs outputs,
    std::size_t row,
    std::size_t sensitivity_index,
    std::size_t sensitivity_count
) {
    constexpr unsigned int price_channels = ComputesCentralPrice ? 1U : 0U;
    constexpr unsigned int channels = 1U + price_channels;
    double sums[channels]{};
    double squares[channels]{};
    for (std::size_t path = threadIdx.x;
         path < launch.paths_per_price;
         path += blockDim.x) {
        const auto result = Policy::template evaluate_path<
            IncludeCentral, IncludePayoff
        >(prepared, key, path);
        if constexpr (ComputesCentralPrice) {
            const double price = static_cast<double>(result.price);
            sums[0U] += price;
            squares[0U] += price * price;
        }
        const double sensitivity = static_cast<double>(result.sensitivity);
        sums[price_channels] += sensitivity;
        squares[price_channels] += sensitivity * sensitivity;
    }
    #pragma unroll
    for (unsigned int channel = 0U; channel < channels; ++channel) {
        const auto moments = reductions::reduce_block(
            sums[channel], squares[channel]
        );
        if (threadIdx.x == 0U) {
            double value = 0.0;
            double error = 0.0;
            reductions::compute_statistics(
                moments,
                launch.paths_per_price,
                value,
                error,
                launch.paths_per_price / blockDim.x
                    + (launch.paths_per_price % blockDim.x != 0U)
            );
            if constexpr (ComputesCentralPrice) {
                if (channel == 0U) {
                    outputs.prices[row] = static_cast<float>(value);
                    outputs.price_standard_errors[row] =
                        static_cast<float>(error);
                } else {
                    const auto index =
                        row * sensitivity_count + sensitivity_index;
                    outputs.gradients[index] = static_cast<float>(value);
                    outputs.gradient_standard_errors[index] =
                        static_cast<float>(error);
                }
            } else {
                const auto index = row * sensitivity_count + sensitivity_index;
                outputs.gradients[index] = static_cast<float>(value);
                outputs.gradient_standard_errors[index] =
                    static_cast<float>(error);
            }
        }
        __syncthreads();
    }
}

template<typename Dynamics, typename ProductPolicy, typename Preparation>
__global__ void first_sensitivity_kernel(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::Outputs outputs,
    DevicePreparedStencilOutputs<3U> stencil_outputs,
    std::uint64_t base_seed
) {
    using Policy = FirstTerminalSensitivityPolicy<
        Dynamics, ProductPolicy, Preparation
    >;
    __shared__ typename Policy::Nodes nodes;
    __shared__ typename Policy::Stencil stencil;
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;
    __shared__ pg::CentralRequirement central_requirement;
    __shared__ int row_valid;

    const std::size_t sensitivity_index = blockIdx.y;
    if (sensitivity_index >= plan.sensitivity_count) return;
    const bool computes_central_price = sensitivity_index == 0U;
    for (std::size_t local_row = blockIdx.x;
         local_row < launch.result_count;
         local_row += gridDim.x) {
        const std::size_t row = launch.result_offset + local_row;
        if (threadIdx.x == 0U) {
            row_valid = prepare_first_row<
                Dynamics, ProductPolicy, Preparation
            >(
                inputs,
                plan,
                row,
                sensitivity_index,
                &nodes,
                &stencil,
                &prepared,
                &central_requirement,
                computes_central_price,
                stencil_outputs
            );
            if (row_valid) key = philox::make_key(base_seed + row);
        }
        __syncthreads();
        if (!row_valid) {
            __syncthreads();
            continue;
        }
        if (computes_central_price) {
            evaluate_first_and_reduce<Policy, true, true, true>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        } else if (central_requirement == pg::CentralRequirement::payoff) {
            evaluate_first_and_reduce<Policy, false, true, true>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        } else if (central_requirement == pg::CentralRequirement::state) {
            evaluate_first_and_reduce<Policy, false, true, false>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        } else {
            evaluate_first_and_reduce<Policy, false, false, false>(
                prepared, key, launch, outputs, row, sensitivity_index,
                plan.sensitivity_count
            );
        }
    }
}

}  // namespace detail

template<typename Dynamics, typename ProductPolicy, typename Preparation>
void launch_device_prepared_terminal_first_sensitivities(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs,
    DevicePreparedStencilOutputs<3U> stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
    if (plan.sensitivity_count == 0U) {
        throw std::invalid_argument(
            "A device-prepared sensitivity launch requires a selection."
        );
    }
    if (configuration.block_count < plan.sensitivity_count) {
        throw std::invalid_argument(
            "The block cap cannot cover every selected sensitivity."
        );
    }
    const std::size_t x_blocks = std::min(
        configuration.result_count,
        configuration.block_count / plan.sensitivity_count
    );
    const std::size_t shared = 2U
        * (configuration.threads_per_block / 32U) * sizeof(double);
    const auto function = detail::first_sensitivity_kernel<
        Dynamics, ProductPolicy, Preparation
    >;
    const dim3 grid(
        static_cast<unsigned int>(x_blocks),
        static_cast<unsigned int>(plan.sensitivity_count)
    );
    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        variant,
        function,
        grid,
        dim3(configuration.threads_per_block),
        shared
    );
    function<<<grid, configuration.threads_per_block, shared>>>(
        inputs,
        plan,
        configuration,
        outputs,
        stencil_outputs,
        configuration.base_seed
    );
    check_cuda(cudaGetLastError(), kernel_name);
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
