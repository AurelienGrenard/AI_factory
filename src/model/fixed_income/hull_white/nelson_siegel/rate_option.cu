// Closed-form Hull-White rate_option pricing fitted to Nelson-Siegel curves.
#include "model/fixed_income/hull_white/nelson_siegel/rate_option.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"

// Include analytics so NVCC can inline the complete pricing formula.
#include "model/fixed_income/hull_white/nelson_siegel/analytics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::hull_white::nelson_siegel {
namespace {

// Prepared identity: the rate-option price is a scaled opposite-side ZCB option.
struct PreparedRow {
    HullWhiteFittedParameters model;
    float bond_option_scale;
    float bond_strike;
    float fixing_time;
    float payment_time;
};

// Prepare the strike and scale shared by the caplet and floorlet transformations.
__device__ __forceinline__ PreparedRow prepare_row(
    const ModelParameters& model,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve,
    const product::RateOptionParameters& product,
    float day_fraction
) {
    const float strike_factor = fmaf(
        static_cast<float>(product.accrual_period) * day_fraction,
        product.strike, 1.0f
    );
    return {
        compose_model(model, initial_curve),
        product.notional * strike_factor,
        1.0f / strike_factor,
        static_cast<float>(product.fixing_time) * day_fraction,
        static_cast<float>(product.payment_time) * day_fraction,
    };
}

// Apply the compile-time-selected rate-option transformation at time zero.
template<OptionSide Side>
__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
    if constexpr (Side == OptionSide::call)
    return row.bond_option_scale * zero_coupon_bond_put_price(
        row.model,
        0.0f,
        0.0f,
        row.fixing_time,
        row.payment_time,
        row.bond_strike
    );
    else
    return row.bond_option_scale * zero_coupon_bond_call_price(
        row.model,
        0.0f,
        0.0f,
        row.fixing_time,
        row.payment_time,
        row.bond_strike
    );
}

// Price one independent row per CUDA thread with coalesced array access.
template<OptionSide Side>
__global__ void hull_white_nelson_siegel_rate_option_kernel(
    const ModelParameters* __restrict__ models,
    const curve::nelson_siegel::NelsonSiegelParameters* __restrict__ curves,
    const product::RateOptionParameters* __restrict__ products,
    std::size_t curve_count,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    float* __restrict__ prices
) {
    const std::size_t launch_index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (launch_index >= launch_result_count) return;

    const std::size_t result_index = result_offset + launch_index;
    std::size_t model_index = result_index;
    std::size_t curve_index = result_index;
    std::size_t product_index = result_index;
    if (cartesian_product) {
        const std::size_t curve_product_count = curve_count * product_count;
        model_index = result_index / curve_product_count;
        const std::size_t remainder = result_index % curve_product_count;
        curve_index = remainder / product_count;
        product_index = remainder % product_count;
    }

    const PreparedRow row = prepare_row(
        models[model_index], curves[curve_index], products[product_index], day_fraction
    );
    prices[result_index] = evaluate_price<Side>(row);
}

// Compose the common checks required by this analytical launcher.
void validate_hull_white_nelson_siegel_rate_option_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::RateOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    const float* device_prices
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_curves, "device_curves");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_model_curve_product_construction(
        model_count,
        curve_count,
        product_count,
        cartesian_product,
        result_count
    );
    validate_day_fraction(day_fraction);
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The Hull-White rate_option launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The Hull-White rate_option thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The Hull-White rate_option launch requires one thread per price."
        );
    }
}

}  // namespace

// Validate and launch the analytical kernel on caller-owned device arrays.
template<OptionSide Side>
void launch_hull_white_nelson_siegel_rate_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::RateOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    validate_hull_white_nelson_siegel_rate_option_launch(
        device_models,
        model_count,
        device_curves,
        curve_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        day_fraction,
        threads_per_block,
        block_count,
        device_prices
    );

    // Launch one analytical rate_option price per active CUDA thread.
    report_cuda_kernel_launch_if_enabled(
        "hull_white.nelson_siegel.rate_option",
        option_side_name(Side),
        hull_white_nelson_siegel_rate_option_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    hull_white_nelson_siegel_rate_option_kernel<Side><<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models,
        device_curves,
        device_products,
        curve_count,
        product_count,
        cartesian_product,
        result_offset,
        launch_result_count,
        day_fraction,
        device_prices
    );
    check_cuda(cudaGetLastError(), "Hull-White rate_option kernel");
}

// Build both public payoff specializations in this CUDA translation unit.
using LaunchSignature = decltype(launch_hull_white_nelson_siegel_rate_option_cuda<OptionSide::call>);
namespace {
[[maybe_unused]] LaunchSignature* launch_instantiation_0 =
    &launch_hull_white_nelson_siegel_rate_option_cuda<OptionSide::call>;
}  // namespace
namespace {
[[maybe_unused]] LaunchSignature* launch_instantiation_1 =
    &launch_hull_white_nelson_siegel_rate_option_cuda<OptionSide::put>;
}  // namespace

}  // namespace ai_factory::workbench::model::hull_white::nelson_siegel
