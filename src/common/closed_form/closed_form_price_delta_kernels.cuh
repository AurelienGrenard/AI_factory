// Grid-stride price/spot-delta execution using an existing closed-form evaluator.
#pragma once

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/equity/price_delta/closed_form_policy.cuh"

namespace ai_factory::workbench::closed_form {

template<ClosedFormPricingPolicy Policy>
__global__ void closed_form_price_delta_kernel(
    typename Policy::DeviceInputs inputs, std::size_t result_offset,
    std::size_t launch_result_count, typename Policy::TimeConfiguration time,
    float* __restrict__ prices, float* __restrict__ deltas
) {
    static_assert(sizeof(typename Policy::PreparedRow) <= kMaximumThreadPreparedRowBytes,
        "Closed-form delta exceeds the 256-byte thread budget; use sequential preparation.");
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < launch_result_count; index += stride) {
        const auto row = inputs.template prepare_row<Policy>(result_offset + index, time);
        const auto value = Policy::evaluate_price_delta(row);
        prices[result_offset + index] = value.price;
        deltas[result_offset + index] = value.delta;
    }
}

template<ClosedFormPricingPolicy Policy>
inline void launch_closed_form_price_delta_cuda(
    const typename Policy::DeviceInputs& inputs,
    const typename Policy::ModelParameters* host_models,
    std::size_t result_count, std::size_t result_offset, std::size_t launch_result_count,
    const typename Policy::TimeConfiguration& time, unsigned int threads_per_block,
    std::size_t block_count, float* prices, float* deltas,
    const char* diagnostic_name, const char* diagnostic_variant
) {
    validate_closed_form_launch<Policy>(inputs, result_count, result_offset,
        launch_result_count, time, threads_per_block, block_count, prices);
    validate_device_pointer(deltas, "device_deltas");
    if (host_models == nullptr) throw std::invalid_argument("Spot delta requires host model mirrors.");
    for (std::size_t model = 0; model < inputs.primary.model_count; ++model) {
        equity::price_delta::validate_spot_bump(host_models[model].spot, inputs.context);
    }
    const auto kernel = closed_form_price_delta_kernel<Policy>;
    int active_blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, threads_per_block, 0U), diagnostic_name);
    if (active_blocks == 0) throw std::invalid_argument("Closed-form delta has no resident block.");
    report_cuda_kernel_launch_if_enabled(diagnostic_name, diagnostic_variant, kernel,
        dim3(static_cast<unsigned int>(block_count)), dim3(threads_per_block), 0U);
    closed_form_price_delta_kernel<Policy><<<static_cast<unsigned int>(block_count), threads_per_block>>>(
        inputs, result_offset, launch_result_count, time, prices, deltas);
    check_cuda(cudaGetLastError(), diagnostic_name);
}

}  // namespace ai_factory::workbench::closed_form
