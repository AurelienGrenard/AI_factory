// Closed-form Black-Scholes Range-accrual kernel.
#include "model/equity/black_scholes/range_accrual.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/normal_distribution.cuh"
#include "common/result_index.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ai_factory::workbench::black_scholes {
namespace {

struct PreparedRow {
    float log_lower_barrier;
    float log_upper_barrier;
    float log_drift_rate;
    float volatility;
    float observation_interval;
    float maturity_discount;
    float discounted_coupon_per_observation;
    std::uint32_t observation_count;
};

__device__ __forceinline__ PreparedRow prepare_row(
    const ModelParameters& model,
    const product::RangeAccrualParameters& product,
    float day_fraction
) {
    const float variance = model.volatility * model.volatility;
    const float observation_years =
        static_cast<float>(product.observation_interval) * day_fraction;
    const float maturity_years =
        static_cast<float>(product.maturity) * day_fraction;
    const float maturity_discount =
        expf(-model.risk_free_rate * maturity_years);
    return {
        logf(product.lower_barrier),
        logf(product.upper_barrier),
        model.risk_free_rate - model.dividend_yield - 0.5f * variance,
        model.volatility,
        observation_years,
        maturity_discount,
        maturity_discount * product.coupon_rate * observation_years,
        product.maturity / product.observation_interval,
    };
}

__device__ __forceinline__ float evaluate_price(const PreparedRow& row) {
    double probability_sum = 0.0;
    for (std::uint32_t observation = 1U;
         observation <= row.observation_count;
         ++observation) {
        const float time =
            static_cast<float>(observation) * row.observation_interval;
        const float standard_deviation = row.volatility * sqrtf(time);
        const float mean = row.log_drift_rate * time;
        const float lower = (row.log_lower_barrier - mean) / standard_deviation;
        const float upper = (row.log_upper_barrier - mean) / standard_deviation;
        probability_sum += static_cast<double>(
            normal_cdf(upper) - normal_cdf(lower)
        );
    }
    return fmaf(
        row.discounted_coupon_per_observation,
        static_cast<float>(probability_sum),
        row.maturity_discount
    );
}

__global__ void black_scholes_range_accrual_kernel(
    const ModelParameters* __restrict__ models,
    const product::RangeAccrualParameters* __restrict__ products,
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
        models[indices.model_index], products[indices.product_index],
        day_fraction
    );
    prices[result_index] = evaluate_price(row);
}

void validate_black_scholes_range_accrual_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
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
            "The Black-Scholes Range-accrual launch batch exceeds the result array."
        );
    }
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    const std::size_t thread_count = checked_workspace_product(
        block_count,
        static_cast<std::size_t>(threads_per_block),
        "The Black-Scholes Range-accrual thread count exceeds size_t."
    );
    if (thread_count < launch_result_count) {
        throw std::invalid_argument(
            "The Black-Scholes Range-accrual launch requires one thread per price."
        );
    }
}

}  // namespace

void launch_black_scholes_range_accrual_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
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
    validate_black_scholes_range_accrual_launch(
        device_models, model_count, device_products, product_count,
        cartesian_product, result_count, result_offset, launch_result_count,
        day_fraction, threads_per_block, block_count, device_prices
    );
    report_cuda_kernel_launch_if_enabled(
        "black_scholes.range_accrual",
        "none",
        black_scholes_range_accrual_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    black_scholes_range_accrual_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, device_products, product_count, cartesian_product,
        result_offset, launch_result_count, day_fraction, device_prices
    );
    check_cuda(cudaGetLastError(), "Black-Scholes Range-accrual kernel");
}

}  // namespace ai_factory::workbench::black_scholes
