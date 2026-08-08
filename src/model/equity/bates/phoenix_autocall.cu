// Bates Phoenix-autocall kernel with fused Philox simulation and payoff reduction.
#include "model/equity/bates/phoenix_autocall.cuh"

#include "common/check_cuda.cuh"
#include "common/result_index.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/reductions.cuh"

// Include the dynamics implementation so NVCC can inline each time step.
#include "model/equity/bates/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::bates {
namespace {

// -------------------------- Phoenix-autocall payoff ---------------------------

// Prepared model and payoff constants shared by one result block.
struct PreparedRow {
    BatesQeParameters model;
    philox::PhiloxKey key;
    float autocall_barrier;
    float coupon_barrier;
    float protection_barrier;
    float coupon_per_observation;
    float discount_per_observation;
    std::uint32_t observation_count;
    std::uint32_t steps_per_observation;
};

// Precompute the model coefficients and payoff constants shared by one block.
__device__ __forceinline__ PreparedRow prepare_row(
    const BatesModelParameters& model,
    const product::PhoenixAutocallParameters& product,
    std::uint32_t observation_count,
    std::uint32_t steps_per_observation,
    std::uint64_t seed
) {
    return {
        prepare_model(
            model,
            product.observation_interval,
            steps_per_observation
        ),
        philox::make_key(seed),
        product.autocall_barrier,
        product.coupon_barrier,
        product.protection_barrier,
        product.annual_coupon_rate * product.observation_interval,
        expf(-model.risk_free_rate * product.observation_interval),
        observation_count,
        steps_per_observation,
    };
}

// Simulate and discount one path without writing it to global memory.
__device__ __forceinline__ float evaluate_path(
    const PreparedRow& row,
    std::size_t path
) {
    BatesState state = initial_state(row.model);
    philox::UniformSequence uniforms(
        row.key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    float present_value = 0.0f;
    float discount = 1.0f;

    for (std::uint32_t observation = 0U;
         observation < row.observation_count;
         ++observation) {
        simulate_interval(
            row.model,
            row.steps_per_observation,
            uniforms,
            normal_cache,
            state
        );
        discount *= row.discount_per_observation;
        const float spot = expf(state.log_spot);
        const float coupon = spot >= row.coupon_barrier
            ? row.coupon_per_observation
            : 0.0f;
        const bool maturity = observation + 1U == row.observation_count;
        if (!maturity && spot >= row.autocall_barrier) {
            return present_value + discount * (1.0f + coupon);
        }
        if (maturity) {
            const float capital = spot >= row.protection_barrier ? 1.0f : spot;
            return present_value + discount * (capital + coupon);
        }
        present_value = fmaf(discount, coupon, present_value);
    }
    return present_value;
}

// Price rows through a bounded persistent grid and write FP32 result moments.
__global__ void bates_phoenix_autocall_kernel(
    const BatesModelParameters* __restrict__ models,
    const product::PhoenixAutocallParameters* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
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
            const product::PhoenixAutocallParameters product =
                products[product_index];
            const std::uint32_t observation_count =
                static_cast<std::uint32_t>(floorf(
                    product.maturity / product.observation_interval + 0.5f
                ));
            const std::uint32_t steps_per_observation =
                static_cast<std::uint32_t>(fmaxf(
                    1.0f,
                    floorf(product.observation_interval / target_dt + 0.5f)
                ));
            prepared = prepare_row(
                models[model_index],
                product,
                observation_count,
                steps_per_observation,
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
            const float payoff = evaluate_path(prepared, path);
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
void validate_bates_phoenix_autocall_launch(
    const BatesModelParameters* device_models,
    std::size_t model_count,
    const product::PhoenixAutocallParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
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
            "The Bates Phoenix-autocall launch batch exceeds the result array."
        );
    }

    // Monte Carlo paths and the requested simulation step must be valid.
    validate_monte_carlo_parameters(
        monte_carlo_paths_per_price, target_dt
    );

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
void launch_bates_phoenix_autocall_cuda(
    const BatesModelParameters* device_models,
    std::size_t model_count,
    const product::PhoenixAutocallParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    validate_bates_phoenix_autocall_launch(
        device_models,
        model_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        target_dt,
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
            bates_phoenix_autocall_kernel,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        "Bates Phoenix autocall occupancy check"
    );
    if (active_blocks_per_multiprocessor == 0) {
        throw std::invalid_argument(
            "Bates Phoenix autocall kernel cannot launch one block per SM."
        );
    }

    // Launch the Bates Phoenix-autocall kernel.
    report_cuda_kernel_launch_if_enabled(
        "bates.phoenix_autocall",
        "default",
        bates_phoenix_autocall_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    bates_phoenix_autocall_kernel<<<
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
        target_dt,
        base_seed,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "Bates Phoenix autocall kernel");
}

}  // namespace ai_factory::workbench::bates
