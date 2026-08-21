// Merton Forward-start option kernel with fused Philox simulation and payoff reduction.
#include "model/equity/merton/forward_start_option.cuh"

#include "common/check_cuda.cuh"
#include "common/result_index.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/reductions.cuh"

// Include the dynamics implementation so NVCC can inline each time step.
#include "model/equity/merton/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::merton {
namespace {

// -------------------------- Forward-start option payoff ---------------------------

// Prepared model and payoff constants shared by one result block.
struct PreparedRow {
    PreparedModel model;
    PreparedTransition transitions[2];
    philox::PhiloxKey key;
    float moneyness;
    float discount;
};

// Precompute the model coefficients and payoff constants shared by one block.
__device__ __forceinline__ PreparedRow prepare_row(
    const ModelParameters& model,
    const product::ForwardStartOptionParameters& product,
    float day_fraction,
    std::uint64_t seed
) {
    const float reset_time =
        static_cast<float>(product.reset_time) * day_fraction;
    const float remaining_time = static_cast<float>(
        product.maturity - product.reset_time
    ) * day_fraction;
    const float maturity =
        static_cast<float>(product.maturity) * day_fraction;
    const PreparedModel prepared_model = prepare_model(model);
    return {
        prepared_model,
        {
            prepare_transition(prepared_model, reset_time),
            prepare_transition(prepared_model, remaining_time),
        },
        philox::make_key(seed),
        product.moneyness,
        expf(-model.risk_free_rate * maturity),
    };
}

// Simulate and discount one path without writing it to global memory.
template<OptionSide Side>
__device__ __forceinline__ float evaluate_path(
    const PreparedRow& row,
    std::size_t path
) {
    float reset_spot = 0.0f;
    const State terminal = simulate_on_calendar(
        row.model,
        row.transitions,
        2U,
        row.key,
        path,
        1U,
        &reset_spot
    );
    const float terminal_spot = expf(terminal.log_spot);
    if constexpr (Side == OptionSide::call)
        return row.discount
            * fmaxf(terminal_spot - row.moneyness * reset_spot, 0.0f);
    else
        return row.discount
            * fmaxf(row.moneyness * reset_spot - terminal_spot, 0.0f);
}

// Price rows through a bounded persistent grid and write FP32 result moments.
template<OptionSide Side>
__global__ void merton_forward_start_option_kernel(
    const ModelParameters* __restrict__ models,
    const product::ForwardStartOptionParameters* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    std::uint64_t base_seed,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    __shared__ PreparedRow prepared;

    for (std::size_t launch_index = blockIdx.x;
         launch_index < launch_result_count;
         launch_index += gridDim.x) {
        const std::size_t result_index = result_offset + launch_index;
        // Thread 0 maps and prepares the next row once for the whole block.
        if (threadIdx.x == 0U) {
            const ModelProductIndices indices =
                decode_model_product_result_index(
                    result_index, product_count, cartesian_product
                );
            const std::size_t model_index = indices.model_index;
            const std::size_t product_index = indices.product_index;
            const product::ForwardStartOptionParameters product =
                products[product_index];
            prepared = prepare_row(
                models[model_index],
                product,
                day_fraction,
                base_seed + result_index
            );
        }
        __syncthreads();

        double sum = 0.0;
        double sumsq = 0.0;
        // Distribute Monte Carlo paths across the threads of this block.
        for (std::size_t path = threadIdx.x;
             path < monte_carlo_paths_per_price;
             path += blockDim.x) {
            const float payoff = evaluate_path<Side>(prepared, path);
            const double value = static_cast<double>(payoff);
            sum += value;
            sumsq += value * value;
        }

        // Reduce all thread-local payoff moments to one block result.
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
            standard_errors[result_index] =
                static_cast<float>(standard_error);
        }
        // No thread may read the old prepared row while thread 0 replaces it.
        __syncthreads();
    }
}

// Compose the common checks required by this specific model/product launcher.
void validate_merton_forward_start_option_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::ForwardStartOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    // All four arrays must already exist in device global memory.
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(device_standard_errors, "device_standard_errors");

    // Input counts must match the aligned or Cartesian result construction.
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The Merton forward-start option batch exceeds the result array."
        );
    }

    // The Monte Carlo path count must be valid.
    validate_monte_carlo_path_count(monte_carlo_paths_per_price);
    validate_day_fraction(day_fraction);

    // The block must fit the GPU and contain a whole number of warps.
    validate_reduction_block_size(threads_per_block);

    // The persistent grid must contain valid blocks and fit gridDim.x.
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);

    // Every result row must receive a distinct uint64_t seed without overflow.
    validate_row_seed_range(result_count, base_seed);
}

}  // namespace

// Validate and launch the pricing kernel on caller-owned device arrays.
template<OptionSide Side>
void launch_merton_forward_start_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::ForwardStartOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    validate_merton_forward_start_option_launch(
        device_models,
        model_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        day_fraction,
        threads_per_block,
        block_count,
        base_seed,
        device_prices,
        device_standard_errors
    );

    // Store one sum and squared sum per warp in shared memory.
    const std::size_t shared_bytes =
        2U * (threads_per_block / 32U) * sizeof(double);

    // Verify that at least one configured block can reside on an SM.
    int active_blocks_per_multiprocessor = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_multiprocessor,
            merton_forward_start_option_kernel<Side>,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        "Merton Forward-start option occupancy check"
    );
    if (active_blocks_per_multiprocessor == 0) {
        throw std::invalid_argument(
            "Merton Forward-start option kernel cannot launch one block per SM."
        );
    }

    // Launch the Merton Forward-start option kernel.
    report_cuda_kernel_launch_if_enabled(
        "merton.forward_start_option",
        option_side_name(Side),
        merton_forward_start_option_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    merton_forward_start_option_kernel<Side><<<
        static_cast<unsigned int>(block_count),
        threads_per_block,
        shared_bytes
    >>>(
        device_models,
        device_products,
        product_count,
        cartesian_product,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        day_fraction,
        base_seed,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "Merton Forward-start option kernel");
}

// Build both public payoff specializations in this CUDA translation unit.
using LaunchSignature = decltype(launch_merton_forward_start_option_cuda<OptionSide::call>);
template LaunchSignature launch_merton_forward_start_option_cuda<OptionSide::call>;
template LaunchSignature launch_merton_forward_start_option_cuda<OptionSide::put>;

}  // namespace ai_factory::workbench::merton
