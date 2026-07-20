// Specialized CUDA implementation of Heston European-call pricing.
// One block owns one model/product result row; its threads fuse path simulation
// with payoff evaluation, then cooperatively reduce the Monte Carlo moments.
#include "heston/european_call.hpp"

#include "common/check_cuda.cuh"
#include "common/reductions.cuh"
#include "heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>

namespace ai_factory::workbench::heston {
namespace {

// This product-level structure combines prepared model dynamics with the two
// quantities used only by the call payoff.
struct PreparedRow {
    HestonQeParameters model;
    float strike;
    float discount;
};

// Precompute the model coefficients and payoff constants shared by one block.
__device__ __forceinline__ PreparedRow prepare_row(
    const HestonModelInput& model,
    const products::EuropeanCallInput& product,
    std::size_t num_steps,
    std::uint64_t seed
) {
    const float maturity = product.maturity;
    return {
        prepare_model(model, maturity, num_steps, seed),
        product.strike,
        expf(-model.risk_free_rate * maturity),
    };
}

// A thread calls this function for each path assigned to it. Every path remains
// in registers; only its scalar discounted payoff returns to the caller.
__device__ __forceinline__ float evaluate_path(
    const PreparedRow& row,
    std::size_t path,
    std::size_t num_steps
) {
    const float terminal =
        simulate_terminal_spot(row.model, path, num_steps);
    return row.discount * fmaxf(terminal - row.strike, 0.0f);
}

// Price one result row per block and write its FP32 price and standard error.
__global__ void heston_european_call_kernel(
    const HestonModelInput* models,
    const products::EuropeanCallInput* products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    std::uint64_t base_seed,
    float* prices,
    float* standard_errors
) {
    // The block index is the logical result-row index in both constructions.
    const std::size_t result_index = blockIdx.x;
    if (result_index >= result_count) return;

    // Aligned rows share one index; Cartesian rows use model-major ordering.
    const std::size_t model_index = cartesian_product
        ? result_index / product_count
        : result_index;
    const std::size_t product_index = cartesian_product
        ? result_index % product_count
        : result_index;
    const products::EuropeanCallInput product = products[product_index];
    const std::size_t num_steps = static_cast<std::size_t>(
        fmaxf(1.0f, floorf(product.maturity / target_dt + 0.5f))
    );
    const std::uint64_t seed = base_seed + result_index;

    // Every thread in the block uses the same row coefficients. Preparing
    // once in shared memory avoids repeating expensive coefficient arithmetic.
    __shared__ PreparedRow prepared;
    if (threadIdx.x == 0U) {
        prepared = prepare_row(
            models[model_index], product, num_steps, seed
        );
    }
    __syncthreads();

    double sum = 0.0;
    double sumsq = 0.0;
    // One block owns one dataset row. Its threads cooperatively cover every
    // path by advancing with the block width, while simulation and payoff stay
    // fused and no path or partial-moment array reaches global memory.
    for (std::size_t path = threadIdx.x;
         path < monte_carlo_paths_per_price;
         path += blockDim.x) {
        const float payoff = evaluate_path(prepared, path, num_steps);
        const double value = static_cast<double>(payoff);
        sum += value;
        sumsq += value * value;
    }

    // Collapse thread-local moments into one pair for the entire result row.
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
        standard_errors[result_index] = static_cast<float>(standard_error);
    }
}

// Compose the common checks required by this specific model/product launcher.
void validate_heston_european_call_launch(
    const HestonModelInput* device_models,
    std::size_t model_count,
    const products::EuropeanCallInput* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(device_standard_errors, "device_standard_errors");
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    validate_monte_carlo_parameters(
        monte_carlo_paths_per_price, target_dt
    );
    validate_warp_aligned_block_size(threads_per_block);
    validate_one_block_per_result_grid(result_count);
    validate_row_seed_range(result_count, base_seed);
}

}  // namespace

// Validate and launch the pricing kernel on caller-owned device arrays.
void launch_heston_european_call_cuda(
    const HestonModelInput* device_models,
    std::size_t model_count,
    const products::EuropeanCallInput* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    validate_heston_european_call_launch(
        device_models,
        model_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        monte_carlo_paths_per_price,
        target_dt,
        threads_per_block,
        base_seed,
        device_prices,
        device_standard_errors
    );

    // Reduction storage holds one sum and one squared sum per warp.
    const std::size_t shared_bytes =
        2U * (threads_per_block / 32U) * sizeof(double);
    heston_european_call_kernel<<<
        static_cast<unsigned int>(result_count),
        threads_per_block,
        shared_bytes
    >>>(
        device_models,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        monte_carlo_paths_per_price,
        target_dt,
        base_seed,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "Heston European call kernel");
}

}  // namespace ai_factory::workbench::heston
