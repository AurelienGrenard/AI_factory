// One block per (row, sensitivity) for gradients and diagonal Hessians.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/equity/price_gradients/device_preparation.cuh"
#include "common/philox.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/launch.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include "common/price_gradients/sensitivity_outputs.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::equity::price_gradients::device_preparation;

namespace diagonal_detail {

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation>
struct TerminalSensitivityPolicy {
    static_assert(pg::requests_second_v<Orders>);
    static constexpr std::size_t kNodeCapacity = 4U;
    using Scenario = typename Preparation::Scenario;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;
    using Nodes = pg::SensitivityNodes<Scenario, kNodeCapacity>;

    struct PreparedRow {
        typename Dynamics::Prepared dynamics[kNodeCapacity];
        typename ProductPolicy::PreparedProduct products[kNodeCapacity];
        const Nodes* nodes;
        const Stencil* stencil;
        std::uint32_t maximum_steps;
        std::uint8_t node_count;

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

    __device__ __forceinline__ static float finalize(
        const PreparedRow& row,
        unsigned int node,
        const typename Dynamics::State& state
    ) {
        const auto handler = ProductPolicy::make_handler(row.products[node]);
        const typename Observation::State terminal{
            Dynamics::spot(state), row.input(node).spot_scale
        };
        return ProductPolicy::template finalize<Observation>(
            row.products[node], terminal, handler
        );
    }

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time
    ) {
        PreparedRow row{};
        row.nodes = nodes;
        row.stencil = stencil;
        row.node_count = static_cast<std::uint8_t>(
            pg::active_node_count(*stencil)
        );
        #pragma unroll
        for (unsigned int node = 0U; node < kNodeCapacity; ++node) {
            if (node >= row.node_count) continue;
            const auto& input = row.input(node);
            auto model = input.model;
            model.spot = input.simulation_spot;
            const float horizon = Dynamics::kExactTerminal
                ? input.maturity_years
                : time.dt;
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
        float first;
        float second;
    };

    __device__ __forceinline__ static PathResult evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        typename Dynamics::State states[kNodeCapacity];
        #pragma unroll
        for (unsigned int node = 0U; node < kNodeCapacity; ++node) {
            if (node < row.node_count) {
                states[node] = Dynamics::initial(row.dynamics[node]);
            }
        }
        typename Dynamics::RandomContext random(key, path);
        simulate_coupled_terminal_nodes<kNodeCapacity, Dynamics>(
            random,
            row.dynamics,
            row.node_count,
            row.maximum_steps,
            [&](unsigned int node) {
                return node == 0U || !row.input(node).reuse_central;
            },
            [&](unsigned int node) { return row.input(node).step_count; },
            [&](unsigned int node) {
                return row.input(node).normal_weights;
            },
            [&](unsigned int node) -> typename Dynamics::State& {
                return states[node];
            }
        );

        pg::SensitivityValues<kNodeCapacity> values{};
        pg::SensitivityValues<kNodeCapacity> terminal_spots{};
        terminal_spots[0U] = Dynamics::spot(states[0U]);
        values[0U] = finalize(row, 0U, states[0U]);
        #pragma unroll
        for (unsigned int node = 1U; node < kNodeCapacity; ++node) {
            if (node < row.node_count) {
                const auto& terminal_state = row.input(node).reuse_central
                    ? states[0U]
                    : states[node];
                terminal_spots[node] = Dynamics::spot(terminal_state);
                values[node] = finalize(row, node, terminal_state);
            }
        }
        auto sensitivity = pg::reconstruct_sensitivity<Orders>(
            *row.stencil, values
        );
        if constexpr (pg::requests_first_v<Orders>) {
            if (row.stencil->kind == pg::StencilKind::centered) {
                sensitivity.first =
                    ProductPolicy::template centered_difference<Observation>(
                        row.products[1U],
                        row.products[2U],
                        {
                            terminal_spots[1U],
                            row.input(1U).spot_scale,
                        },
                        {
                            terminal_spots[2U],
                            row.input(2U).spot_scale,
                        },
                        row.stencil->represented_width
                    );
            }
        }
        return {values[0U], sensitivity.first, sensitivity.second};
    }
};

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation>
__device__ __noinline__ bool prepare_row(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    std::size_t row,
    std::size_t sensitivity_index,
    typename TerminalSensitivityPolicy<
        Orders, Dynamics, ProductPolicy, Preparation
    >::Nodes* nodes,
    typename TerminalSensitivityPolicy<
        Orders, Dynamics, ProductPolicy, Preparation
    >::Stencil* stencil,
    typename TerminalSensitivityPolicy<
        Orders, Dynamics, ProductPolicy, Preparation
    >::PreparedRow* prepared,
    DevicePreparedStencilOutputs<4U> stencil_outputs
) {
    using Policy = TerminalSensitivityPolicy<
        Orders, Dynamics, ProductPolicy, Preparation
    >;
    const auto indices = pg::price_row_indices(
        row, plan.construction, plan.product_count
    );
    int preparation_error = preparation::valid;
    typename Preparation::Scenario central{};
    bool valid_row = Preparation::make_central(
        inputs.models[indices.model],
        inputs.products[indices.product],
        plan.time,
        central
    );
    pg::SensitivityTask<typename Preparation::Scenario, 4U> task{};
    if (valid_row) {
        valid_row = preparation::build_sensitivity_task<Orders, Preparation>(
            central,
            inputs.sensitivities[sensitivity_index],
            plan.time,
            task,
            preparation_error
        );
    } else {
        preparation_error = preparation::invalid_central;
    }
    if (valid_row) {
        *nodes = task.nodes;
        *stencil = task.stencil;
        *prepared = Policy::prepare(nodes, stencil, plan.time);
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
    return valid_row;
}

template<typename Policy, pg::SensitivityOrders Orders, bool OwnsPrice>
__device__ __forceinline__ void evaluate_and_reduce(
    const typename Policy::PreparedRow& prepared,
    philox::PhiloxKey key,
    const pg::LaunchConfiguration& launch,
    pg::SensitivityOutputs outputs,
    std::size_t row,
    std::size_t sensitivity_index,
    std::size_t sensitivity_count
) {
    constexpr unsigned int price_channels = OwnsPrice ? 1U : 0U;
    constexpr unsigned int first_channels =
        pg::requests_first_v<Orders> ? 1U : 0U;
    constexpr unsigned int second_channels =
        pg::requests_second_v<Orders> ? 1U : 0U;
    constexpr unsigned int channel_count =
        price_channels + first_channels + second_channels;
    double sums[channel_count]{};
    double squares[channel_count]{};
    for (std::size_t path = threadIdx.x;
         path < launch.paths_per_price;
         path += blockDim.x) {
        const auto result = Policy::evaluate_path(prepared, key, path);
        unsigned int channel = 0U;
        if constexpr (OwnsPrice) {
            const double price = static_cast<double>(result.price);
            sums[channel] += price;
            squares[channel] += price * price;
            ++channel;
        }
        if constexpr (pg::requests_first_v<Orders>) {
            const double first = static_cast<double>(result.first);
            sums[channel] += first;
            squares[channel] += first * first;
            ++channel;
        }
        if constexpr (pg::requests_second_v<Orders>) {
            const double second = static_cast<double>(result.second);
            sums[channel] += second;
            squares[channel] += second * second;
        }
    }
    #pragma unroll
    for (unsigned int channel = 0U;
         channel < channel_count;
         ++channel) {
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
            unsigned int output_channel = 0U;
            if constexpr (OwnsPrice) {
                if (channel == output_channel) {
                    outputs.prices[row] = static_cast<float>(value);
                    outputs.price_standard_errors[row] =
                        static_cast<float>(error);
                }
                ++output_channel;
            }
            const auto index = row * sensitivity_count + sensitivity_index;
            if constexpr (pg::requests_first_v<Orders>) {
                if (channel == output_channel) {
                    outputs.gradients[index] = static_cast<float>(value);
                    outputs.gradient_standard_errors[index] =
                        static_cast<float>(error);
                }
                ++output_channel;
            }
            if constexpr (pg::requests_second_v<Orders>) {
                if (channel == output_channel) {
                    outputs.diagonal_hessians[index] =
                        static_cast<float>(value);
                    outputs.diagonal_hessian_standard_errors[index] =
                        static_cast<float>(error);
                }
            }
        }
        __syncthreads();
    }
}

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation>
__global__ void sensitivity_kernel(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    std::uint64_t base_seed
) {
    using Policy = TerminalSensitivityPolicy<
        Orders, Dynamics, ProductPolicy, Preparation
    >;
    __shared__ typename Policy::Nodes nodes;
    __shared__ typename Policy::Stencil stencil;
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;
    __shared__ bool valid_row;

    const std::size_t sensitivity_index = blockIdx.y;
    if (sensitivity_index >= plan.sensitivity_count) return;
    const bool owns_price = sensitivity_index == 0U;
    for (std::size_t local_row = blockIdx.x;
         local_row < launch.result_count;
         local_row += gridDim.x) {
        const std::size_t row = launch.result_offset + local_row;
        if (threadIdx.x == 0U) {
            valid_row = prepare_row<
                Orders, Dynamics, ProductPolicy, Preparation
            >(
                inputs,
                plan,
                row,
                sensitivity_index,
                &nodes,
                &stencil,
                &prepared,
                stencil_outputs
            );
            if (valid_row) key = philox::make_key(base_seed + row);
        }
        __syncthreads();
        if (!valid_row) {
            __syncthreads();
            continue;
        }
        if (owns_price) {
            evaluate_and_reduce<Policy, Orders, true>(
                prepared,
                key,
                launch,
                outputs,
                row,
                sensitivity_index,
                plan.sensitivity_count
            );
        } else {
            evaluate_and_reduce<Policy, Orders, false>(
                prepared,
                key,
                launch,
                outputs,
                row,
                sensitivity_index,
                plan.sensitivity_count
            );
        }
    }
}

}  // namespace diagonal_detail

template<
    pg::SensitivityOrders Orders,
    typename Dynamics,
    typename ProductPolicy,
    typename Preparation>
void launch_device_prepared_terminal_diagonal_sensitivities(
    DevicePreparedInputs<Preparation> inputs,
    DevicePreparedPlan plan,
    const pg::LaunchConfiguration& configuration,
    pg::SensitivityOutputs outputs,
    DevicePreparedStencilOutputs<4U> stencil_outputs,
    const char* kernel_name,
    const char* variant
) {
    static_assert(pg::requests_second_v<Orders>);
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
    const auto function = diagonal_detail::sensitivity_kernel<
        Orders, Dynamics, ProductPolicy, Preparation
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
