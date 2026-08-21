// Closed-form European swaptions under the Ornstein-Uhlenbeck short rate.
#include "model/fixed_income/ornstein_uhlenbeck/european_swaption.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/result_index.cuh"

// Include analytics so NVCC can inline the Jamshidian decomposition.
#include "model/fixed_income/ornstein_uhlenbeck/analytics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {
namespace {

// Model, contract scalars, and schedule views consumed by one pricing thread.
struct PreparedRow {
    ModelParameters model;
    float notional;
    float strike;
    float exercise_time;
    const std::uint32_t* payment_times;
    const std::uint32_t* accrual_periods;
    float day_fraction;
    std::uint32_t payment_count;
};

// Prepare one physical-settlement swaption whose swap starts at exercise.
__device__ __forceinline__ PreparedRow prepare_row(
    const ModelParameters& model,
    const product::EuropeanSwaptionParameters& product,
    float day_fraction
) {
    return {
        model,
        product.notional,
        product.strike,
        static_cast<float>(product.exercise_time) * day_fraction,
        product.payment_times,
        product.accrual_periods,
        day_fraction,
        product.payment_count,
    };
}

// Map call to payer and put to receiver without a runtime payoff branch.
template<OptionSide Side>
__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
    if constexpr (Side == OptionSide::call)
        return row.notional * european_payer_swaption_price(
            row.model,
            row.model.initial_state,
            0.0f,
            row.exercise_time,
            row.strike,
            row.payment_times,
            row.accrual_periods,
            row.day_fraction,
            row.payment_count
        );
    else
        return row.notional * european_receiver_swaption_price(
            row.model,
            row.model.initial_state,
            0.0f,
            row.exercise_time,
            row.strike,
            row.payment_times,
            row.accrual_periods,
            row.day_fraction,
            row.payment_count
        );
}

// Price one independent row per CUDA thread with model-major indexing.
template<OptionSide Side>
__global__ void ornstein_uhlenbeck_european_swaption_kernel(
    const ModelParameters* __restrict__ models,
    const product::EuropeanSwaptionParameters* __restrict__ products,
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
    const ModelProductIndices indices = decode_model_product_result_index(
        result_index, product_count, cartesian_product
    );
    const PreparedRow row = prepare_row(
        models[indices.model_index],
        products[indices.product_index],
        day_fraction
    );
    prices[result_index] = evaluate_price<Side>(row);
}

// Compose the common checks required by this analytical launcher.
void validate_ornstein_uhlenbeck_european_swaption_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanSwaptionParameters* device_products,
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
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    validate_day_fraction(day_fraction);
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The OU European swaption launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The OU European swaption thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The OU European swaption launch requires one thread per price."
        );
    }
}

}  // namespace

// Validate and launch the analytical kernel on caller-owned device arrays.
template<OptionSide Side>
void launch_ornstein_uhlenbeck_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanSwaptionParameters* device_products,
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
    validate_ornstein_uhlenbeck_european_swaption_launch(
        device_models,
        model_count,
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

    report_cuda_kernel_launch_if_enabled(
        "ornstein_uhlenbeck.european_swaption",
        option_side_name(Side),
        ornstein_uhlenbeck_european_swaption_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    ornstein_uhlenbeck_european_swaption_kernel<Side><<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models,
        device_products,
        product_count,
        cartesian_product,
        result_offset,
        launch_result_count,
        day_fraction,
        device_prices
    );
    check_cuda(cudaGetLastError(), "OU European swaption kernel");
}

// Build both public payer and receiver specializations in this translation unit.
using LaunchSignature = decltype(
    launch_ornstein_uhlenbeck_european_swaption_cuda<OptionSide::call>
);
template LaunchSignature
launch_ornstein_uhlenbeck_european_swaption_cuda<OptionSide::call>;
template LaunchSignature
launch_ornstein_uhlenbeck_european_swaption_cuda<OptionSide::put>;

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
