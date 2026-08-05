// Closed-form rate_option pricing under the Ornstein-Uhlenbeck short-rate model.
#include "model/ornstein_uhlenbeck/rate_option.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"

// Include analytics so NVCC can inline the complete pricing formula.
#include "model/ornstein_uhlenbeck/analytics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {
namespace {

// Prepared identity: the rate-option price is a scaled opposite-side ZCB option.
struct PreparedRow {
    OrnsteinUhlenbeckModelParameters model;
    float bond_option_scale;
    float bond_strike;
    float fixing_time;
    float payment_time;
};

// Prepare the strike and scale shared by the caplet and floorlet transformations.
__device__ __forceinline__ PreparedRow prepare_row(
    const OrnsteinUhlenbeckModelParameters& model,
    const product::RateOptionParameters& product
) {
    const float strike_factor = fmaf(
        product.accrual_period, product.strike, 1.0f
    );
    return {
        model,
        product.notional * strike_factor,
        1.0f / strike_factor,
        product.fixing_time,
        product.payment_time,
    };
}

// Apply the compile-time-selected rate-option transformation at time zero.
template<OptionSide Side>
__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
    if constexpr (Side == OptionSide::call)
    return row.bond_option_scale * zero_coupon_bond_put_price(
        row.model,
        row.model.initial_state,
        0.0f,
        row.fixing_time,
        row.payment_time,
        row.bond_strike
    );
    else
    return row.bond_option_scale * zero_coupon_bond_call_price(
        row.model,
        row.model.initial_state,
        0.0f,
        row.fixing_time,
        row.payment_time,
        row.bond_strike
    );
}

// Price one independent row per CUDA thread with coalesced array access.
template<OptionSide Side>
__global__ void ornstein_uhlenbeck_rate_option_kernel(
    const OrnsteinUhlenbeckModelParameters* __restrict__ models,
    const product::RateOptionParameters* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float* __restrict__ prices
) {
    const std::size_t launch_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (launch_index >= launch_result_count) return;

    const std::size_t result_index = result_offset + launch_index;
    const std::size_t model_index = cartesian_product
        ? result_index / product_count
        : result_index;
    const std::size_t product_index = cartesian_product
        ? result_index % product_count
        : result_index;
    const PreparedRow row = prepare_row(
        models[model_index], products[product_index]
    );
    prices[result_index] = evaluate_price<Side>(row);
}

// Compose the common checks required by this analytical launcher.
void validate_ornstein_uhlenbeck_rate_option_launch(
    const OrnsteinUhlenbeckModelParameters* device_models,
    std::size_t model_count,
    const product::RateOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    const float* device_prices
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The Ornstein-Uhlenbeck rate_option launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The Ornstein-Uhlenbeck rate_option thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The Ornstein-Uhlenbeck rate_option launch requires one thread per price."
        );
    }
}

}  // namespace

// Validate and launch the analytical kernel on caller-owned device arrays.
template<OptionSide Side>
void launch_ornstein_uhlenbeck_rate_option_cuda(
    const OrnsteinUhlenbeckModelParameters* device_models,
    std::size_t model_count,
    const product::RateOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    validate_ornstein_uhlenbeck_rate_option_launch(
        device_models,
        model_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        threads_per_block,
        block_count,
        device_prices
    );

    // Launch one analytical rate_option price per active CUDA thread.
    report_cuda_kernel_launch_if_enabled(
        "ornstein_uhlenbeck.rate_option",
        option_side_name(Side),
        ornstein_uhlenbeck_rate_option_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    ornstein_uhlenbeck_rate_option_kernel<Side><<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models,
        device_products,
        product_count,
        cartesian_product,
        result_offset,
        launch_result_count,
        device_prices
    );
    check_cuda(cudaGetLastError(), "Ornstein-Uhlenbeck rate_option kernel");
}

// Build both public payoff specializations in this CUDA translation unit.
using LaunchSignature = decltype(launch_ornstein_uhlenbeck_rate_option_cuda<OptionSide::call>);
template LaunchSignature launch_ornstein_uhlenbeck_rate_option_cuda<OptionSide::call>;
template LaunchSignature launch_ornstein_uhlenbeck_rate_option_cuda<OptionSide::put>;

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
