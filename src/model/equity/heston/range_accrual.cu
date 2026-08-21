// Heston Range Accrual kernel with fused Philox simulation and payoff reduction.
#include "model/equity/heston/range_accrual.cuh"

#include "common/check_cuda.cuh"
#include "common/result_index.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/reductions.cuh"

// Include the dynamics implementation so NVCC can inline each time step.
#include "model/equity/heston/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::heston {
namespace {

// ------------------------- Range Accrual payoff -------------------------

// Prepared model and payoff constants shared by one result block.
struct PreparedRow {
    PreparedModel model;
    philox::PhiloxKey key;
    float lower_log_spot;
    float upper_log_spot;
    float maturity_discount;
    float discounted_coupon_per_observation;
    std::uint32_t observation_count;
    std::uint32_t steps_per_observation;
};

// Precompute the model coefficients and payoff constants shared by one block.
__device__ __forceinline__ PreparedRow prepare_row(
    const ModelParameters& model,
    const product::RangeAccrualParameters& product,
    float dt,
    std::uint32_t simulation_steps_per_day,
    std::uint64_t seed
) {
    const std::uint32_t observation_count =
        product.maturity / product.observation_interval;
    const std::uint32_t steps_per_observation =
        simulation_steps_per_day * product.observation_interval;
    const float observation_years =
        static_cast<float>(steps_per_observation) * dt;
    const float maturity_years = static_cast<float>(
        observation_count * steps_per_observation
    ) * dt;
    const PreparedModel prepared_model = prepare_model(model, dt);
    const float maturity_discount = expf(
        -model.risk_free_rate * maturity_years
    );
    return {
        prepared_model,
        philox::make_key(seed),
        prepared_model.initial_log_spot + logf(product.lower_barrier),
        prepared_model.initial_log_spot + logf(product.upper_barrier),
        maturity_discount,
        maturity_discount
            * product.coupon_rate
            * observation_years,
        observation_count,
        steps_per_observation,
    };
}

// Simulate and discount one path without writing it to global memory.
__device__ __forceinline__ float evaluate_path(
    const PreparedRow& row,
    std::size_t path
) {
    State state = initial_state(row.model);
    philox::UniformSequence uniforms(
        row.key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    std::uint32_t in_range_count = 0U;

    for (std::uint32_t observation = 0U;
         observation < row.observation_count;
         ++observation) {
        for (std::uint32_t step = 0U;
             step < row.steps_per_observation;
             ++step) {
            simulate_one_step(row.model, uniforms, normal_cache, state);
        }
        in_range_count += static_cast<std::uint32_t>(
            state.log_spot >= row.lower_log_spot
            && state.log_spot <= row.upper_log_spot
        );
    }
    return fmaf(
        row.discounted_coupon_per_observation,
        static_cast<float>(in_range_count),
        row.maturity_discount
    );
}

// Price rows through a bounded persistent grid and write FP32 result moments.
__global__ void heston_range_accrual_kernel(
    const ModelParameters* __restrict__ models,
    const product::RangeAccrualParameters* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
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
            const product::RangeAccrualParameters product =
                products[product_index];
            prepared = prepare_row(
                models[model_index],
                product,
                dt,
                simulation_steps_per_day,
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
void validate_heston_range_accrual_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
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
            "The Heston Range Accrual launch batch exceeds the result array."
        );
    }

    // Monte Carlo paths and the requested simulation step must be valid.
    validate_monte_carlo_parameters(
        monte_carlo_paths_per_price, dt
    );

    // The block must fit the GPU and contain a whole number of warps.
    validate_simulation_steps_per_day(simulation_steps_per_day);
    validate_reduction_block_size(threads_per_block);

    // The persistent grid must contain valid blocks and fit gridDim.x.
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);

    // Every result row must receive a distinct uint64_t seed without overflow.
    validate_row_seed_range(result_count, base_seed);
}

}  // namespace

// Validate and launch the pricing kernel on caller-owned device arrays.
void launch_heston_range_accrual_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    validate_heston_range_accrual_launch(
        device_models,
        model_count,
        device_products,
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        monte_carlo_paths_per_price,
        dt,
        simulation_steps_per_day,
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
            heston_range_accrual_kernel,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        "Heston Range Accrual occupancy check"
    );
    if (active_blocks_per_multiprocessor == 0) {
        throw std::invalid_argument(
            "Heston Range Accrual kernel cannot launch one block per SM."
        );
    }

    // Launch the Heston Range Accrual kernel.
    report_cuda_kernel_launch_if_enabled(
        "heston.range_accrual",
        "default",
        heston_range_accrual_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    heston_range_accrual_kernel<<<
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
        dt,
        simulation_steps_per_day,
        base_seed,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "Heston Range Accrual kernel");
}

}  // namespace ai_factory::workbench::heston
