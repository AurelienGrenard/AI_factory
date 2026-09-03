// Generic one-block-per-price CUDA execution for cooperative closed forms.
#pragma once

#include "common/check_cuda.cuh"
#include "common/closed_form/concepts.cuh"
#include "common/cuda_kernel_diagnostics.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::closed_form {

template<CooperativeClosedFormPricingPolicy PricingPolicy>
__global__ void cooperative_closed_form_price_kernel(
    typename PricingPolicy::DeviceInputs inputs,
    std::size_t result_offset,
    std::size_t launch_result_count,
    typename PricingPolicy::TimeConfiguration time_configuration,
    std::uint32_t workspace_capacity,
    float* __restrict__ prices
) {
    static_assert(
        sizeof(typename PricingPolicy::PreparedRow)
            <= kMaximumSharedPreparedRowBytes,
        "Cooperative closed-form PreparedRow exceeds the 256-byte shared "
        "budget; store a compact view over device-resident data."
    );
    __shared__ typename PricingPolicy::PreparedRow prepared;
    extern __shared__ __align__(16) unsigned char workspace_storage[];

    for (std::size_t launch_index = blockIdx.x;
         launch_index < launch_result_count;
         launch_index += gridDim.x) {
        const std::size_t result_index = result_offset + launch_index;
        if (threadIdx.x == 0U) {
            prepared = inputs.template prepare_row<PricingPolicy>(
                result_index,
                time_configuration
            );
        }
        __syncthreads();

        const float price = PricingPolicy::evaluate_price(
            prepared,
            reinterpret_cast<std::byte*>(workspace_storage),
            workspace_capacity
        );
        if (threadIdx.x == 0U) prices[result_index] = price;
        __syncthreads();
    }
}

template<CooperativeClosedFormPricingPolicy PricingPolicy>
inline void validate_cooperative_closed_form_launch(
    const typename PricingPolicy::DeviceInputs& inputs,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    const typename PricingPolicy::TimeConfiguration& time_configuration,
    std::uint32_t workspace_capacity,
    unsigned int threads_per_block,
    std::size_t block_count,
    const float* device_prices
) {
    static_assert(
        sizeof(typename PricingPolicy::PreparedRow)
            <= kMaximumSharedPreparedRowBytes,
        "Cooperative closed-form PreparedRow exceeds the 256-byte shared "
        "budget; store a compact view over device-resident data."
    );
    inputs.validate(result_count);
    validate_time_configuration(time_configuration);
    validate_device_pointer(device_prices, "device_prices");
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The cooperative closed-form launch batch exceeds the result "
            "array."
        );
    }
    if (workspace_capacity == 0U) {
        throw std::invalid_argument(
            "The cooperative closed-form workspace capacity is zero."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
}

template<CooperativeClosedFormPricingPolicy PricingPolicy>
inline bool launch_cooperative_closed_form_cuda(
    const typename PricingPolicy::DeviceInputs& inputs,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    const typename PricingPolicy::TimeConfiguration& time_configuration,
    std::uint32_t workspace_capacity,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name
) {
    validate_cooperative_closed_form_launch<PricingPolicy>(
        inputs,
        result_count,
        result_offset,
        launch_result_count,
        time_configuration,
        workspace_capacity,
        threads_per_block,
        block_count,
        device_prices
    );
    const std::size_t shared_bytes =
        PricingPolicy::required_shared_memory_bytes(workspace_capacity);

    cudaFuncAttributes attributes{};
    check_cuda(
        cudaFuncGetAttributes(
            &attributes,
            cooperative_closed_form_price_kernel<PricingPolicy>
        ),
        operation_name
    );
    if (shared_bytes
        > static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes)) {
        return false;
    }

    int active_blocks_per_multiprocessor = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_multiprocessor,
            cooperative_closed_form_price_kernel<PricingPolicy>,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        operation_name
    );
    if (active_blocks_per_multiprocessor == 0) return false;

    report_cuda_kernel_launch_if_enabled(
        diagnostic_name,
        diagnostic_variant,
        cooperative_closed_form_price_kernel<PricingPolicy>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    cooperative_closed_form_price_kernel<PricingPolicy><<<
        static_cast<unsigned int>(block_count),
        threads_per_block,
        shared_bytes
    >>>(
        inputs,
        result_offset,
        launch_result_count,
        time_configuration,
        workspace_capacity,
        device_prices
    );
    check_cuda(cudaGetLastError(), operation_name);
    return true;
}

}  // namespace ai_factory::workbench::closed_form
