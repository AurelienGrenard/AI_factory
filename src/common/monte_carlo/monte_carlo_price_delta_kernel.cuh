// Price and paired spot-delta moments over the same Philox paths and reduction order.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"
#include "common/monte_carlo/monte_carlo_kernel.cuh"

namespace ai_factory::workbench::monte_carlo {

template<typename Policy>
concept PriceDeltaMonteCarloPolicy = MonteCarloLaunchPolicy<Policy>
    && requires(const typename Policy::PreparedRow& row,
                philox::PhiloxKey key, std::size_t path) {
        { Policy::evaluate_path(row, key, path) }
            -> std::same_as<equity::price_delta::PairedPayoff>;
    };

template<PriceDeltaMonteCarloPolicy Policy>
__global__ void monte_carlo_price_delta_kernel(
    typename Policy::DeviceInputs inputs, std::size_t result_offset,
    std::size_t launch_result_count, std::size_t paths_per_price,
    typename Policy::TimeConfiguration time, std::uint64_t base_seed,
    float* __restrict__ prices, float* __restrict__ price_errors,
    float* __restrict__ deltas, float* __restrict__ delta_errors
) {
    static_assert(sizeof(typename Policy::PreparedRow) <= kMaximumSharedPreparedRowBytes,
                  "Price-delta PreparedRow exceeds the 2048-byte shared budget; use a compact view.");
    __shared__ typename Policy::PreparedRow prepared;
    __shared__ philox::PhiloxKey key;
    for (std::size_t index = blockIdx.x; index < launch_result_count; index += gridDim.x) {
        const std::size_t result = result_offset + index;
        if (threadIdx.x == 0U) {
            prepared = inputs.template prepare_row<Policy>(result, time);
            key = philox::make_key(base_seed + result);
        }
        __syncthreads();
        // FP64 is confined to the existing long Monte Carlo moment contract.
        double sum = 0.0, sumsq = 0.0, delta_sum = 0.0, delta_sumsq = 0.0;
        for (std::size_t path = threadIdx.x; path < paths_per_price; path += blockDim.x) {
            const auto payoff = Policy::evaluate_path(prepared, key, path);
            const double price = static_cast<double>(payoff.price);
            const double delta = static_cast<double>(payoff.delta);
            sum += price;
            sumsq += price * price;
            delta_sum += delta;
            delta_sumsq += delta * delta;
        }
        const auto price_moments = reductions::reduce_block(sum, sumsq);
        if (threadIdx.x == 0U) {
            double mean, error;
            reductions::compute_statistics(price_moments, paths_per_price, mean, error,
                paths_per_price / blockDim.x + (paths_per_price % blockDim.x != 0U));
            prices[result] = static_cast<float>(mean);
            price_errors[result] = static_cast<float>(error);
        }
        __syncthreads();  // The two reductions reuse the same shared scratch.
        const auto delta_moments = reductions::reduce_block(delta_sum, delta_sumsq);
        if (threadIdx.x == 0U) {
            double mean, error;
            reductions::compute_statistics(delta_moments, paths_per_price, mean, error,
                paths_per_price / blockDim.x + (paths_per_price % blockDim.x != 0U));
            deltas[result] = static_cast<float>(mean);
            delta_errors[result] = static_cast<float>(error);
        }
        __syncthreads();
    }
}

template<PriceDeltaMonteCarloPolicy Policy>
inline void launch_monte_carlo_price_delta_cuda(
    const typename Policy::DeviceInputs& inputs,
    const typename Policy::HostInputs& host_inputs,
    std::size_t result_count, std::size_t result_offset,
    std::size_t launch_result_count, std::size_t paths_per_price,
    const typename Policy::TimeConfiguration& time,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t base_seed, float* prices, float* price_errors,
    float* deltas, float* delta_errors,
    const char* diagnostic_name, const char* diagnostic_variant
) {
    validate_monte_carlo_launch<Policy>(inputs, host_inputs, result_count,
        result_offset, launch_result_count, paths_per_price, time,
        threads_per_block, block_count, base_seed, prices, price_errors);
    validate_device_pointer(deltas, "device_deltas");
    validate_device_pointer(delta_errors, "device_delta_standard_errors");
    if (inputs.context.relative_width != host_inputs.bump.relative_width) {
        throw std::invalid_argument("Host/device spot bump configurations differ.");
    }
    const auto kernel = monte_carlo_price_delta_kernel<Policy>;
    const std::size_t shared_bytes = 2U * (threads_per_block / 32U) * sizeof(double);
    int active_blocks = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, threads_per_block, shared_bytes), diagnostic_name);
    if (active_blocks == 0) throw std::invalid_argument("Price-delta kernel has no resident block.");
    report_cuda_kernel_launch_if_enabled(diagnostic_name, diagnostic_variant, kernel,
        dim3(static_cast<unsigned int>(block_count)), dim3(threads_per_block), shared_bytes);
    monte_carlo_price_delta_kernel<Policy><<<
        static_cast<unsigned int>(block_count), threads_per_block, shared_bytes
    >>>(inputs, result_offset, launch_result_count, paths_per_price, time,
        base_seed, prices, price_errors, deltas, delta_errors);
    check_cuda(cudaGetLastError(), diagnostic_name);
}

}  // namespace ai_factory::workbench::monte_carlo
