// Generic grid-stride CUDA execution for independent closed-form prices.
#pragma once

#include "common/check_cuda.cuh"
#include "common/closed_form/concepts.cuh"
#include "common/cuda_kernel_diagnostics.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::closed_form {

template<ClosedFormPricingPolicy PricingPolicy>
__device__ __forceinline__ void price_one(
    const typename PricingPolicy::DeviceInputs& inputs,
    std::size_t result_index,
    const typename PricingPolicy::TimeConfiguration& time_configuration,
    float* __restrict__ prices
) {
    static_assert(
        sizeof(typename PricingPolicy::PreparedRow)
            <= kMaximumThreadPreparedRowBytes,
        "Closed-form PreparedRow exceeds the 256-byte per-thread budget; "
        "store a compact view and keep variable-length data in device memory."
    );
    const typename PricingPolicy::PreparedRow row =
        inputs.template prepare_row<PricingPolicy>(
            result_index,
            time_configuration
        );
    prices[result_index] = PricingPolicy::evaluate_price(row);
}

template<ClosedFormPricingPolicy PricingPolicy, bool GridStride>
__global__ void closed_form_price_kernel(
    typename PricingPolicy::DeviceInputs inputs,
    std::size_t result_offset,
    std::size_t launch_result_count,
    typename PricingPolicy::TimeConfiguration time_configuration,
    float* __restrict__ prices
) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if constexpr (!GridStride) {
        if (thread >= launch_result_count) return;
        price_one<PricingPolicy>(
            inputs,
            result_offset + thread,
            time_configuration,
            prices
        );
    } else {
        const std::size_t stride =
            static_cast<std::size_t>(gridDim.x) * blockDim.x;
        for (std::size_t launch_index = thread;
             launch_index < launch_result_count;
             launch_index += stride) {
            const std::size_t result_index = result_offset + launch_index;
            price_one<PricingPolicy>(
                inputs,
                result_index,
                time_configuration,
                prices
            );
        }
    }
}

template<ClosedFormPricingPolicy PricingPolicy>
inline void validate_closed_form_launch(
    const typename PricingPolicy::DeviceInputs& inputs,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    const typename PricingPolicy::TimeConfiguration& time_configuration,
    unsigned int threads_per_block,
    std::size_t block_count,
    const float* device_prices
) {
    inputs.validate(result_count);
    validate_time_configuration(time_configuration);
    validate_device_pointer(device_prices, "device_prices");
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The closed-form launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
}

template<ClosedFormPricingPolicy PricingPolicy>
inline void launch_closed_form_cuda(
    const typename PricingPolicy::DeviceInputs& inputs,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    const typename PricingPolicy::TimeConfiguration& time_configuration,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name
) {
    validate_closed_form_launch<PricingPolicy>(
        inputs,
        result_count,
        result_offset,
        launch_result_count,
        time_configuration,
        threads_per_block,
        block_count,
        device_prices
    );
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The closed-form thread count exceeds size_t."
    );
    if (thread_count >= launch_result_count) {
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            closed_form_price_kernel<PricingPolicy, false>,
            dim3(static_cast<unsigned int>(block_count)),
            dim3(threads_per_block)
        );
        closed_form_price_kernel<PricingPolicy, false><<<
            static_cast<unsigned int>(block_count),
            threads_per_block
        >>>(
            inputs,
            result_offset,
            launch_result_count,
            time_configuration,
            device_prices
        );
    } else {
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            closed_form_price_kernel<PricingPolicy, true>,
            dim3(static_cast<unsigned int>(block_count)),
            dim3(threads_per_block)
        );
        closed_form_price_kernel<PricingPolicy, true><<<
            static_cast<unsigned int>(block_count),
            threads_per_block
        >>>(
            inputs,
            result_offset,
            launch_result_count,
            time_configuration,
            device_prices
        );
    }
    check_cuda(cudaGetLastError(), operation_name);
}

}  // namespace ai_factory::workbench::closed_form
