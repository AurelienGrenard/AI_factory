// Materialize represented stencils from compact row and sensitivity inputs.
#pragma once

#include "common/cuda_kernel_diagnostics.cuh"
#include "common/price_gradients/device_preparation.cuh"
#include "common/price_gradients/device_prepared_launch.cuh"
#include "common/price_gradients/row_mapping.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

namespace stencil_detail {

template<pg::SensitivityOrders Orders, typename Inputs>
__global__ void prepare_stencils_kernel(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t result_offset,
    std::size_t result_count,
    DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > outputs
) {
    using Preparation = typename Inputs::PreparationPolicy;
    constexpr std::size_t node_capacity =
        pg::SensitivityTraits<Orders>::node_capacity;
    const std::size_t task_count = result_count * plan.sensitivity_count;
    for (std::size_t task_index =
             static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         task_index < task_count;
         task_index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const std::size_t local_row = task_index / plan.sensitivity_count;
        const std::size_t sensitivity_index =
            task_index - local_row * plan.sensitivity_count;
        const std::size_t row = result_offset + local_row;
        int preparation_error = preparation::valid;
        typename Preparation::Scenario central{};
        bool valid_row = inputs.make_central(row, plan, central);
        pg::SensitivityTask<
            typename Preparation::Scenario,
            node_capacity
        > sensitivity_task{};
        if (valid_row) {
            valid_row = preparation::build_sensitivity_task<
                Orders,
                Preparation
            >(
                central,
                inputs.sensitivities[sensitivity_index],
                plan.time,
                sensitivity_task,
                preparation_error
            );
        } else {
            preparation_error = preparation::invalid_central;
        }
        if (valid_row) {
            outputs.stencils[
                row * plan.sensitivity_count + sensitivity_index
            ] = sensitivity_task.stencil;
        } else {
            preparation::record_error(
                outputs.error,
                preparation_error,
                row,
                sensitivity_index
            );
        }
    }
}

}  // namespace stencil_detail

template<pg::SensitivityOrders Orders, typename Inputs>
void launch_device_prepared_stencils(
    Inputs inputs,
    DevicePreparedPlan plan,
    std::size_t result_offset,
    std::size_t result_count,
    DevicePreparedStencilOutputs<
        pg::SensitivityTraits<Orders>::node_capacity
    > outputs,
    const char* kernel_name
) {
    if (result_count == 0U || plan.sensitivity_count == 0U) return;
    if (result_offset >= plan.result_count
        || result_count > plan.result_count - result_offset) {
        throw std::invalid_argument(
            "Device stencil preparation exceeds its row range."
        );
    }
    const std::size_t task_count = result_count * plan.sensitivity_count;
    constexpr unsigned int threads = 256U;
    const auto blocks = static_cast<unsigned int>(std::min<std::size_t>(
        (task_count + threads - 1U) / threads,
        65535U
    ));
    const auto function = stencil_detail::prepare_stencils_kernel<
        Orders,
        Inputs
    >;
    report_cuda_kernel_launch_if_enabled(
        kernel_name,
        "represented_stencils",
        function,
        dim3(blocks),
        dim3(threads),
        0U
    );
    function<<<blocks, threads>>>(
        inputs,
        plan,
        result_offset,
        result_count,
        outputs
    );
    check_cuda(cudaGetLastError(), kernel_name);
}

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
