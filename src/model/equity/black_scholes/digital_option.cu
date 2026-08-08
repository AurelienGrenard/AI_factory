// Closed-form Black-Scholes DigitalOption kernel.
#include "model/equity/black_scholes/digital_option.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/normal_distribution.cuh"
#include "common/result_index.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::black_scholes {
namespace {

struct PreparedRow {
    float discounted_cash_payoff;
    float d2;
};

__device__ __forceinline__ PreparedRow prepare_row(
    const BlackScholesModelParameters& model,
    const product::DigitalOptionParameters& product
) {
    const float sqrt_maturity = sqrtf(product.maturity);
    const float volatility_sqrt_maturity = model.volatility * sqrt_maturity;
    const float variance = model.volatility * model.volatility;
    const float d2 = (
        logf(model.spot / product.strike)
        + (model.risk_free_rate - model.dividend_yield - 0.5f * variance)
            * product.maturity
    ) / volatility_sqrt_maturity;
    return {
        product.cash_payoff * expf(-model.risk_free_rate * product.maturity),
        d2,
    };
}

template<OptionSide Side>
__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
    if constexpr (Side == OptionSide::call) {
        return row.discounted_cash_payoff * normal_cdf(row.d2);
    } else {
        return row.discounted_cash_payoff * normal_cdf(-row.d2);
    }
}

template<OptionSide Side>
__global__ void black_scholes_digital_option_kernel(
    const BlackScholesModelParameters* __restrict__ models,
    const product::DigitalOptionParameters* __restrict__ products,
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
    const ModelProductIndices indices = decode_model_product_result_index(
        result_index, product_count, cartesian_product
    );
    const PreparedRow row = prepare_row(
        models[indices.model_index], products[indices.product_index]
    );
    prices[result_index] = evaluate_price<Side>(row);
}

void validate_black_scholes_digital_option_launch(
    const BlackScholesModelParameters* device_models,
    std::size_t model_count,
    const product::DigitalOptionParameters* device_products,
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
            "The Black-Scholes DigitalOption launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The Black-Scholes DigitalOption thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The Black-Scholes DigitalOption launch requires one thread per price."
        );
    }
}

}  // namespace

template<OptionSide Side>
void launch_black_scholes_digital_option_cuda(
    const BlackScholesModelParameters* device_models,
    std::size_t model_count,
    const product::DigitalOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    validate_black_scholes_digital_option_launch(
        device_models, model_count, device_products, product_count,
        cartesian_product, result_count, result_offset, launch_result_count,
        threads_per_block, block_count, device_prices
    );
    report_cuda_kernel_launch_if_enabled(
        "black_scholes.digital_option",
        option_side_name(Side),
        black_scholes_digital_option_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    black_scholes_digital_option_kernel<Side><<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, device_products, product_count, cartesian_product,
        result_offset, launch_result_count, device_prices
    );
    check_cuda(cudaGetLastError(), "Black-Scholes DigitalOption kernel");
}

using LaunchSignature =
    decltype(launch_black_scholes_digital_option_cuda<OptionSide::call>);
template LaunchSignature
launch_black_scholes_digital_option_cuda<OptionSide::call>;
template LaunchSignature
launch_black_scholes_digital_option_cuda<OptionSide::put>;

}  // namespace ai_factory::workbench::black_scholes
