// Closed-form zero-coupon bond puts under the Ornstein-Uhlenbeck short rate.
#include "model/ornstein_uhlenbeck/zero_coupon_bond_put.cuh"

#include "common/check_cuda.cuh"

// Include analytics so NVCC can inline the complete pricing formula.
#include "model/ornstein_uhlenbeck/analytics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {
namespace {

// Model and contract constants consumed by one pricing thread.
struct PreparedRow {
    OrnsteinUhlenbeckModelParameters model;
    float notional;
    float strike;
    float option_expiry;
    float bond_maturity;
};

// Prepare one put on a zero-coupon bond.
__device__ __forceinline__ PreparedRow prepare_row(
    const OrnsteinUhlenbeckModelParameters& model,
    const product::ZeroCouponBondPutParameters& product
) {
    return {
        model,
        product.notional,
        product.strike,
        product.option_expiry,
        product.bond_maturity,
    };
}

// Apply the zero-coupon bond put formula at time zero.
__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
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
__global__ void ornstein_uhlenbeck_zero_coupon_bond_put_kernel(
    const OrnsteinUhlenbeckModelParameters* __restrict__ models,
    const product::ZeroCouponBondPutParameters* __restrict__ products,
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
    prices[result_index] = evaluate_price(row);
}

// Compose the common checks required by this analytical launcher.
void validate_ornstein_uhlenbeck_zero_coupon_bond_put_launch(
    const OrnsteinUhlenbeckModelParameters* device_models,
    std::size_t model_count,
    const product::ZeroCouponBondPutParameters* device_products,
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
            "The OU zero-coupon bond put launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The OU zero-coupon bond put thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The OU zero-coupon bond put launch requires one thread per price."
        );
    }
}

}  // namespace

// Validate and launch the analytical kernel on caller-owned device arrays.
void launch_ornstein_uhlenbeck_zero_coupon_bond_put_cuda(
    const OrnsteinUhlenbeckModelParameters* device_models,
    std::size_t model_count,
    const product::ZeroCouponBondPutParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    validate_ornstein_uhlenbeck_zero_coupon_bond_put_launch(
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

    // Launch one analytical zero-coupon bond put per active CUDA thread.
    ornstein_uhlenbeck_zero_coupon_bond_put_kernel<<<
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
    check_cuda(cudaGetLastError(), "OU zero-coupon bond put kernel");
}

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
