// Generic one-block-per-price Monte Carlo execution layer.
#pragma once

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/monte_carlo/concepts.cuh"
#include "common/reductions.cuh"
#include "common/result_index.cuh"
#include "common/simulation/schedule.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::monte_carlo {

template<ScalarMonteCarloPricingPolicy Pricing>
__global__ void monte_carlo_price_kernel(
    typename Pricing::DeviceInputs inputs,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    typename Pricing::Schedule::TimeConfiguration time_configuration,
    std::uint64_t base_seed,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    static_assert(
        sizeof(typename Pricing::PreparedRow)
            <= kMaximumSharedPreparedRowBytes,
        "Monte Carlo PreparedRow exceeds the 2048-byte shared-memory "
        "budget; store a compact ScheduleView instead of embedding the "
        "complete runtime calendar."
    );
    __shared__ typename Pricing::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;

    for (std::size_t launch_index = blockIdx.x;
         launch_index < launch_result_count;
         launch_index += gridDim.x) {
        const std::size_t result_index = result_offset + launch_index;
        if (threadIdx.x == 0U) {
            prepared = inputs.template prepare_row<Pricing>(
                result_index,
                time_configuration
            );
            key = philox::make_key(base_seed + result_index);
        }
        __syncthreads();

        double sum = 0.0;
        double sumsq = 0.0;
        for (std::size_t path = threadIdx.x;
             path < monte_carlo_paths_per_price;
            path += blockDim.x) {
            const double value = static_cast<double>(
                Pricing::evaluate_path(prepared, key, path)
            );
            sum += value;
            sumsq += value * value;
        }

        const reductions::MomentSums total =
            reductions::reduce_block(sum, sumsq);
        if (threadIdx.x == 0U) {
            double price = 0.0;
            double standard_error = 0.0;
            reductions::compute_statistics(
                total,
                monte_carlo_paths_per_price,
                price,
                standard_error
            );
            prices[result_index] = static_cast<float>(price);
            standard_errors[result_index] =
                static_cast<float>(standard_error);
        }
        __syncthreads();
    }
}

template<ScalarMonteCarloPricingPolicy Pricing>
inline void validate_monte_carlo_launch(
    const typename Pricing::DeviceInputs& inputs,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    const typename Pricing::Schedule::TimeConfiguration& time_configuration,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    static_assert(
        sizeof(typename Pricing::PreparedRow)
            <= kMaximumSharedPreparedRowBytes,
        "Monte Carlo PreparedRow exceeds the 2048-byte shared-memory "
        "budget; store a compact ScheduleView instead of embedding the "
        "complete runtime calendar."
    );
    inputs.validate(result_count);
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(
        device_standard_errors,
        "device_standard_errors"
    );

    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The Monte Carlo launch batch exceeds the result array."
        );
    }

    validate_monte_carlo_path_count(monte_carlo_paths_per_price);
    simulation::validate_time_configuration(time_configuration);
    validate_reduction_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    validate_row_seed_range(result_count, base_seed);
}

template<ScalarMonteCarloPricingPolicy Pricing>
inline void launch_monte_carlo_cuda(
    const typename Pricing::DeviceInputs& inputs,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    const typename Pricing::Schedule::TimeConfiguration& time_configuration,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name
) {
    validate_monte_carlo_launch<Pricing>(
        inputs,
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        time_configuration,
        threads_per_block,
        block_count,
        base_seed,
        device_prices,
        device_standard_errors
    );

    const std::size_t shared_bytes =
        2U * (threads_per_block / 32U) * sizeof(double);
    int active_blocks_per_multiprocessor = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_multiprocessor,
            monte_carlo_price_kernel<Pricing>,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        operation_name
    );
    if (active_blocks_per_multiprocessor == 0) {
        throw std::invalid_argument(
            "The Monte Carlo kernel cannot launch one block per SM."
        );
    }

    report_cuda_kernel_launch_if_enabled(
        diagnostic_name,
        diagnostic_variant,
        monte_carlo_price_kernel<Pricing>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    monte_carlo_price_kernel<Pricing><<<
        static_cast<unsigned int>(block_count),
        threads_per_block,
        shared_bytes
    >>>(
        inputs,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        time_configuration,
        base_seed,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), operation_name);
}

}  // namespace ai_factory::workbench::monte_carlo
