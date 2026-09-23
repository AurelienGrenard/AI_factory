// One block per terminal price with the central row prepared on device.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/philox.cuh"
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

    struct PreparedRow {
        typename Dynamics::Prepared dynamics;
        typename ProductPolicy::PreparedProduct product;
        Scenario input;
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

    __device__ __forceinline__ static PreparedRow prepare(
        const Scenario& input,
        pg::TimeConfiguration time
    ) {
        auto model = input.model;
        model.spot = input.simulation_spot;
        const float horizon = Dynamics::kExactTerminal
            ? input.maturity_years
            : time.dt;
        return {
            Dynamics::prepare(model, horizon),
            ProductPolicy::prepare_product(
                input.model,
                input.product,
                {
                    static_cast<float>(time.simulation_steps_per_day) * time.dt,
                    input.maturity_years,
                }
            ),
            input,
        };
    }

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        auto state = Dynamics::initial(row.dynamics);
        typename Dynamics::RandomContext random(key, path);
        if constexpr (Dynamics::kExactTerminal) {
            const auto innovations = [&] {
                if constexpr (requires {
                                  Dynamics::draw(random, row.dynamics);
                              }) {
                    return Dynamics::draw(random, row.dynamics);
                } else {
                    return Dynamics::draw(random);
                }
            }();
            Dynamics::transition(
                row.dynamics,
                innovations,
                row.input.normal_weights,
                state
            );
        } else {
            for (std::uint32_t step = 0U;
                 step < row.input.step_count;
                 ++step) {
                const auto innovations = Dynamics::draw(random);
                Dynamics::transition(
                    row.dynamics, innovations, nullptr, state
                );
            }
        }
        const auto handler = ProductPolicy::make_handler(row.product);
        const typename Observation::State terminal{
            Dynamics::spot(state),
            row.input.spot_scale,
        };
        return ProductPolicy::template finalize<Observation>(
            row.product, terminal, handler
        );
    }
};

template<typename Dynamics, typename ProductPolicy, typename Preparation>
__global__ void terminal_price_kernel(
    DevicePreparedInputs<Preparation> inputs,
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
            const auto indices = pg::price_row_indices(
                row, plan.construction, plan.product_count
            );
            typename Preparation::Scenario central{};
            valid_row = Preparation::make_central(
                inputs.models[indices.model],
                inputs.products[indices.product],
                plan.time,
                central
            );
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
        typename HostPlan::Preparation
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
