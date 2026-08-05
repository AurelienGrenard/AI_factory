// Closed-form zero-coupon bond options under the G2 short rate.
#include "model/g2/zero_coupon_bond_option.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"

// Include analytics so NVCC can inline the complete pricing formula.
#include "model/g2/analytics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::g2 {
namespace {

// Model and contract constants consumed by one pricing thread.
struct PreparedRow {
    G2ModelParameters model;
    float notional;
    float strike;
    float option_expiry;
    float bond_maturity;
};

// Prepare one option on a zero-coupon bond.
__device__ __forceinline__ PreparedRow prepare_row(
    const G2ModelParameters& model,
    const product::ZeroCouponBondOptionParameters& product
) {
    return {
        model,
        product.notional,
        product.strike,
        product.option_expiry,
        product.bond_maturity,
    };
}

// Apply the zero-coupon bond option formula at time zero.
template<OptionSide Side>
__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
    if constexpr (Side == OptionSide::call)
    return row.notional * zero_coupon_bond_call_price(
        row.model,
        row.model.initial_state,
        0.0f,
        row.option_expiry,
        row.bond_maturity,
        row.strike
    );
    else
    return row.notional * zero_coupon_bond_put_price(
        row.model,
        row.model.initial_state,
        0.0f,
        row.option_expiry,
        row.bond_maturity,
        row.strike
    );
}

// Price one independent row per CUDA thread with coalesced array access.
template<OptionSide Side>
__global__ void g2_zero_coupon_bond_option_kernel(
    const G2ModelParameters* __restrict__ models,
    const product::ZeroCouponBondOptionParameters* __restrict__ products,
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
void validate_g2_zero_coupon_bond_option_launch(
    const G2ModelParameters* device_models,
    std::size_t model_count,
    const product::ZeroCouponBondOptionParameters* device_products,
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
            "The G2 zero-coupon bond option launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The G2 zero-coupon bond option thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The G2 zero-coupon bond option launch requires one thread per price."
        );
    }
}

}  // namespace

// Validate and launch the analytical kernel on caller-owned device arrays.
template<OptionSide Side>
void launch_g2_zero_coupon_bond_option_cuda(
    const G2ModelParameters* device_models,
    std::size_t model_count,
    const product::ZeroCouponBondOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    validate_g2_zero_coupon_bond_option_launch(
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

    // Launch one analytical zero-coupon bond option per active CUDA thread.
    report_cuda_kernel_launch_if_enabled(
        "g2.zero_coupon_bond_option",
        option_side_name(Side),
        g2_zero_coupon_bond_option_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    g2_zero_coupon_bond_option_kernel<Side><<<
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
    check_cuda(cudaGetLastError(), "G2 zero-coupon bond option kernel");
}

// Build both public payoff specializations in this CUDA translation unit.
using LaunchSignature = decltype(launch_g2_zero_coupon_bond_option_cuda<OptionSide::call>);
template LaunchSignature launch_g2_zero_coupon_bond_option_cuda<OptionSide::call>;
template LaunchSignature launch_g2_zero_coupon_bond_option_cuda<OptionSide::put>;

}  // namespace ai_factory::workbench::model::g2
