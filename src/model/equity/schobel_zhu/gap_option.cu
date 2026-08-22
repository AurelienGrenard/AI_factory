// Schobel-Zhu gap-option kernel with fused Philox simulation and payoff reduction.
#include "model/equity/schobel_zhu/gap_option.cuh"

#include "common/check_cuda.cuh"
#include "common/result_index.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/reductions.cuh"

// Include the dynamics implementation so NVCC can inline each time step.
#include "model/equity/schobel_zhu/dynamics.cu"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ai_factory::workbench::schobel_zhu {
namespace {

// ----------------------------- Gap payoff --------------------------------

// Prepared model and payoff constants shared by one result block.
struct PreparedRow {
    PreparedModel model;
    philox::PhiloxKey key;
    float trigger_strike;
    float payoff_strike;
    float discount;
    std::uint32_t num_steps;
};

// Precompute the model coefficients and payoff constants shared by one block.
__device__ __forceinline__ PreparedRow prepare_row(
    const ModelParameters& model,
    const product::GapOptionParameters& product,
    float dt,
    std::uint32_t simulation_steps_per_day,
    std::uint64_t seed
) {
    const std::uint32_t num_steps =
        simulation_steps_per_day * product.maturity;
    const float maturity_years = static_cast<float>(num_steps) * dt;
    return {
        prepare_model(model, dt),
        philox::make_key(seed),
        product.trigger_strike,
        product.payoff_strike,
        expf(-model.risk_free_rate * maturity_years),
        num_steps,
    };
}

// Simulate and discount one path without writing it to global memory.
template<OptionSide Side>
__device__ __forceinline__ float evaluate_path(
    const PreparedRow& row,
    std::size_t path
) {
    const State terminal =
        simulate_terminal_state(row.model, row.key, path, row.num_steps);
    const float terminal_spot = expf(terminal.log_spot);
    if constexpr (Side == OptionSide::call) {
        const bool pays = terminal_spot > row.trigger_strike;
        const float payoff = terminal_spot - row.payoff_strike;
        return pays ? row.discount * payoff : 0.0f;
    } else {
        const bool pays = terminal_spot < row.trigger_strike;
        const float payoff = row.payoff_strike - terminal_spot;
        return pays ? row.discount * payoff : 0.0f;
    }
}

// Price rows through a bounded persistent grid and write FP32 result moments.
template<OptionSide Side>
__global__ void schobel_zhu_gap_option_kernel(
    const ModelParameters* __restrict__ models,
    const product::GapOptionParameters* __restrict__ products,
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
            const product::GapOptionParameters product =
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
void validate_schobel_zhu_gap_option_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::GapOptionParameters* device_products,
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
            "The Schobel-Zhu gap-option launch batch exceeds the result array."
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
template<OptionSide Side>
void launch_schobel_zhu_gap_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::GapOptionParameters* device_products,
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
    validate_schobel_zhu_gap_option_launch(
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
            schobel_zhu_gap_option_kernel<Side>,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        "Schobel-Zhu Gap option occupancy check"
    );
    if (active_blocks_per_multiprocessor == 0) {
        throw std::invalid_argument(
            "Schobel-Zhu Gap option kernel cannot launch one block per SM."
        );
    }

    // Launch the Schobel-Zhu Gap-option kernel.
    report_cuda_kernel_launch_if_enabled(
        "schobel_zhu.gap_option",
        option_side_name(Side),
        schobel_zhu_gap_option_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    schobel_zhu_gap_option_kernel<Side><<<
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
    check_cuda(cudaGetLastError(), "Schobel-Zhu Gap option kernel");
}

// Build both public payoff specializations in this CUDA translation unit.
using LaunchSignature = decltype(launch_schobel_zhu_gap_option_cuda<OptionSide::call>);
namespace {
[[maybe_unused]] LaunchSignature* launch_instantiation_0 =
    &launch_schobel_zhu_gap_option_cuda<OptionSide::call>;
}  // namespace
namespace {
[[maybe_unused]] LaunchSignature* launch_instantiation_1 =
    &launch_schobel_zhu_gap_option_cuda<OptionSide::put>;
}  // namespace

}  // namespace ai_factory::workbench::schobel_zhu
