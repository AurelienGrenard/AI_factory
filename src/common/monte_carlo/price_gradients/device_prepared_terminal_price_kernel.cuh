// One block per terminal price with the central row prepared on device.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/philox.cuh"
#include "common/monte_carlo/price_gradients/coupled_terminal_simulation.cuh"
#include "common/monte_carlo/price_gradients/terminal_sensitivity_policy.cuh"
#include "common/price_gradients/device_prepared_validation.hpp"
#include "common/price_gradients/row_mapping.cuh"
#include "common/reductions.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::equity::price_gradients::device_preparation;

namespace detail {

template<typename Dynamics, typename ProductPolicy, typename Preparation>
struct CentralTerminalPolicy {
    using Scenario = typename Preparation::Scenario;
    using NodePolicy = SelectedTerminalNodePolicy<
        Dynamics, ProductPolicy, Preparation
    >;

    struct PreparedRow {
        typename Dynamics::Prepared dynamics;
        typename NodePolicy::Metadata metadata;
        Scenario input;
    };

    __device__ __forceinline__ static PreparedRow prepare(
        const Scenario& input,
        pg::TimeConfiguration time
    ) {
        return {
            NodePolicy::prepare_dynamics(input, time),
            NodePolicy::prepare_metadata(input, time),
            input,
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        typename Dynamics::Prepared prepared[1U]{row.dynamics};
        typename Dynamics::State states[1U]{
            Dynamics::initial(row.dynamics)
        };
        typename Dynamics::RandomContext random(key, path);
        simulate_coupled_terminal_nodes<1U, Dynamics>(
            random,
            prepared,
            1U,
            row.input.step_count,
            [](unsigned int) { return true; },
            [&](unsigned int) { return row.input.step_count; },
            [&](unsigned int) { return row.input.normal_weights; },
            [&](unsigned int node) -> typename Dynamics::State& {
                return states[node];
            }
        );
        return NodePolicy::payoff(
            row.metadata, NodePolicy::observe(states[0U])
        );
    }
};

template<typename Dynamics, typename ProductPolicy, typename Preparation, typename Inputs>
__global__ void terminal_price_kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    pg::LaunchConfiguration launch,
    pg::Outputs outputs,
    DevicePreparedStencilOutputs<3U> preparation_outputs,
    std::uint64_t base_seed
) {
    using Policy = CentralTerminalPolicy<
        Dynamics, ProductPolicy, Preparation
    >;
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;
    __shared__ int valid_row;
    for (std::size_t local_row = blockIdx.x;
         local_row < launch.result_count;
         local_row += gridDim.x) {
        const std::size_t row = launch.result_offset + local_row;
        if (threadIdx.x == 0U) {
            typename Preparation::Scenario central{};
            valid_row = inputs.make_central(row, plan, central);
            if (valid_row) {
                prepared = Policy::prepare(central, plan.time);
                key = philox::make_key(base_seed + row);
            } else {
                preparation::record_error(
                    preparation_outputs.error,
                    preparation::invalid_central,
                    row,
                    0U
                );
            }
        }
        __syncthreads();
        if (!valid_row) {
            __syncthreads();
            continue;
        }
        double sum = 0.0;
        double square = 0.0;
        for (std::size_t path = threadIdx.x;
             path < launch.paths_per_price;
             path += blockDim.x) {
            const double value = static_cast<double>(
                Policy::evaluate_path(prepared, key, path)
            );
            sum += value;
            square += value * value;
        }
        const auto moments = reductions::reduce_block(sum, square);
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
            outputs.prices[row] = static_cast<float>(value);
            outputs.price_standard_errors[row] = static_cast<float>(error);
        }
        __syncthreads();
    }
}

}  // namespace detail

template<typename Dynamics, typename ProductPolicy, typename HostPlan>
void launch_device_prepared_terminal_prices(
    const HostPlan& host,
    typename HostPlan::DeviceInputs device,
    typename HostPlan::StencilOutputs preparation_outputs,
    const pg::LaunchConfiguration& configuration,
    pg::Outputs outputs,
    const char* kernel_name,
    const char* variant
) {
    const auto plan = make_device_prepared_plan(host);
    const std::vector<pg::BufferRange> output_ranges{
        pg::checked_buffer_range(
            outputs.prices, host.result_count, sizeof(float)
        ),
        pg::checked_buffer_range(
            outputs.price_standard_errors,
            host.result_count,
            sizeof(float)
        ),
    };
    validate_device_prepared_launch(
        device,
        plan,
        preparation_outputs,
        configuration,
        output_ranges
    );
    const auto function = detail::terminal_price_kernel<
        Dynamics,
        ProductPolicy,
        typename HostPlan::Preparation,
        typename HostPlan::DeviceInputs
    >;
    const std::size_t shared = 2U
        * (configuration.threads_per_block / 32U) * sizeof(double);
    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        variant,
        function,
        dim3(static_cast<unsigned int>(configuration.block_count)),
        dim3(configuration.threads_per_block),
        shared
    );
    function<<<configuration.block_count, configuration.threads_per_block, shared>>>(
        device,
        plan,
        configuration,
        outputs,
        preparation_outputs,
        configuration.base_seed
    );
    check_cuda(cudaGetLastError(), kernel_name);
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
