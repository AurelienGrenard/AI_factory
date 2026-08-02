// Closed-form G2++ floorlet pricing fitted to Nelson-Siegel curves.
#include "model/g2_plus_plus/nelson_siegel/floorlet.cuh"

#include "common/check_cuda.cuh"

// Include analytics so NVCC can inline the complete pricing formula.
#include "model/g2_plus_plus/nelson_siegel/analytics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel {
namespace {

// Prepared identity: floorlet price = bond_option_scale * ZCB call price.
struct PreparedRow {
    G2PlusPlusFittedParameters model;
    float bond_option_scale;
    float bond_strike;
    float fixing_time;
    float payment_time;
};

// Prepare the zero-coupon call representation of one floorlet.
__device__ __forceinline__ PreparedRow prepare_row(
    const G2PlusPlusModelParameters& model,
    const curve::nelson_siegel::NelsonSiegelParameters& initial_curve,
    const product::FloorletParameters& product
) {
    const float strike_factor = fmaf(
        product.accrual_period, product.strike, 1.0f
    );
    return {
        compose_model(model, initial_curve),
        product.notional * strike_factor,
        1.0f / strike_factor,
        product.fixing_time,
        product.payment_time,
    };
}

// Apply the floorlet-to-zero-coupon-call identity at time zero.
__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
    return row.bond_option_scale * zero_coupon_bond_call_price(
        row.model,
        model::g2::G2State{0.0f, 0.0f},
        0.0f,
        row.fixing_time,
        row.payment_time,
        row.bond_strike
    );
}

// Price one independent row per CUDA thread with coalesced array access.
__global__ void g2_plus_plus_nelson_siegel_floorlet_kernel(
    const G2PlusPlusModelParameters* __restrict__ models,
    const curve::nelson_siegel::NelsonSiegelParameters* __restrict__ curves,
    const product::FloorletParameters* __restrict__ products,
    std::size_t curve_count,
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
        models[model_index], curves[curve_index], products[product_index]
    );
    prices[result_index] = evaluate_price(row);
}

// Compose the common checks required by this analytical launcher.
void validate_g2_plus_plus_nelson_siegel_floorlet_launch(
    const G2PlusPlusModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::FloorletParameters* device_products,
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
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The G2++ floorlet launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The G2++ floorlet thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The G2++ floorlet launch requires one thread per price."
        );
    }
}

}  // namespace

// Validate and launch the analytical kernel on caller-owned device arrays.
void launch_g2_plus_plus_nelson_siegel_floorlet_cuda(
    const G2PlusPlusModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::FloorletParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    validate_g2_plus_plus_nelson_siegel_floorlet_launch(
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
        threads_per_block,
        block_count,
        device_prices
    );

    // Launch one analytical floorlet price per active CUDA thread.
    g2_plus_plus_nelson_siegel_floorlet_kernel<<<
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
        device_prices
    );
    check_cuda(cudaGetLastError(), "G2++ floorlet kernel");
}

}  // namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel
