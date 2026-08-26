// Rough-Bergomi European-option kernel with explicit hybrid-history lanes.
#include "model/equity/rough_bergomi/european_option.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/reductions.cuh"
#include "common/result_index.cuh"

// Include the dynamics implementation so NVCC can inline the convolution.
#include "model/equity/rough_bergomi/dynamics.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::model::equity::rough_bergomi {
namespace {

// Prepared model and payoff constants shared by one result block.
struct PreparedRow {
    RoughBergomiPreparedParameters model;
    philox::PhiloxKey key;
    float strike;
    float discount;
    std::uint32_t num_steps;
};

std::size_t checked_product(
    std::size_t left,
    std::size_t right,
    const char* description
) {
    if (right != 0U
        && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::overflow_error(description);
    }
    return left * right;
}

std::size_t rounded_step_count(float maturity, float target_dt) {
    return static_cast<std::size_t>(
        std::fmax(1.0f, std::floor(maturity / target_dt + 0.5f))
    );
}

std::size_t reduction_shared_bytes(unsigned int threads_per_block) {
    return 2U * (threads_per_block / 32U) * sizeof(double);
}

std::size_t hybrid_grid_shared_bytes(std::size_t maximum_step_count) {
    return checked_product(
        2U * sizeof(float),
        maximum_step_count,
        "rough-Bergomi shared-grid byte count overflows size_t."
    );
}

// Precompute the model coefficients and payoff constants shared by one block.
__device__ __forceinline__ PreparedRow prepare_row(
    const RoughBergomiModelParameters& model,
    const product::EuropeanOptionParameters& product,
    float day_fraction,
    std::uint32_t num_steps,
    std::uint64_t seed
) {
    const float maturity = static_cast<float>(product.maturity) * day_fraction;
    return {
        prepare_model(model, maturity, num_steps),
        philox::make_key(seed),
        product.strike,
        expf(-model.risk_free_rate * maturity),
        num_steps,
    };
}

// Simulate and discount one path while reusing the calling thread's lane.
template<OptionSide Side>
__device__ __forceinline__ float evaluate_path(
    const PreparedRow& row,
    const RoughBergomiHybridGridView& grid,
    RoughBergomiHistoryView history,
    std::size_t path
) {
    const RoughBergomiState terminal = simulate_terminal_state(
        row.model,
        grid,
        history,
        row.key,
        path,
        row.num_steps
    );
    const float terminal_spot = expf(terminal.log_spot);
    if constexpr (Side == OptionSide::call)
        return row.discount * fmaxf(terminal_spot - row.strike, 0.0f);
    else
        return row.discount * fmaxf(row.strike - terminal_spot, 0.0f);
}

// Price rows through a bounded persistent grid and write FP32 result moments.
template<OptionSide Side>
__global__ void rough_bergomi_european_option_kernel(
    const RoughBergomiModelParameters* __restrict__ models,
    const product::EuropeanOptionParameters* __restrict__ products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    float target_dt,
    std::size_t maximum_step_count,
    float* __restrict__ history_workspace,
    std::uint64_t base_seed,
    float* __restrict__ prices,
    float* __restrict__ standard_errors
) {
    __shared__ PreparedRow prepared;
    __shared__ std::size_t prepared_step_count;
    extern __shared__ double dynamic_shared[];
    const std::size_t warp_count = blockDim.x / 32U;
    float* const far_weights = reinterpret_cast<float*>(
        dynamic_shared + 2U * warp_count
    );
    float* const log_variance_corrections =
        far_weights + maximum_step_count;
    float* const block_history = history_workspace
        + static_cast<std::size_t>(blockIdx.x)
            * maximum_step_count * blockDim.x;
    const RoughBergomiHistoryView history = {
        block_history + threadIdx.x,
        blockDim.x,
    };

    for (std::size_t launch_index = blockIdx.x;
         launch_index < launch_result_count;
         launch_index += gridDim.x) {
        const std::size_t result_index = result_offset + launch_index;
        if (threadIdx.x == 0U) {
            const ModelProductIndices indices =
                decode_model_product_result_index(
                    result_index, product_count, cartesian_product
                );
            const product::EuropeanOptionParameters product =
                products[indices.product_index];
            const float maturity =
                static_cast<float>(product.maturity) * day_fraction;
            prepared_step_count = static_cast<std::size_t>(
                fmaxf(1.0f, floorf(maturity / target_dt + 0.5f))
            );
            prepared = prepare_row(
                models[indices.model_index],
                product,
                day_fraction,
                static_cast<std::uint32_t>(prepared_step_count),
                base_seed + result_index
            );
        }
        __syncthreads();

        // Protect the caller-owned buffers if its plan does not cover a row.
        if (prepared_step_count > maximum_step_count) {
            if (threadIdx.x == 0U) {
                const float invalid = __int_as_float(0x7fffffff);
                prices[result_index] = invalid;
                standard_errors[result_index] = invalid;
            }
            __syncthreads();
            continue;
        }

        prepare_hybrid_grid(
            prepared.model,
            static_cast<std::uint32_t>(prepared.num_steps),
            far_weights,
            log_variance_corrections,
            threadIdx.x,
            blockDim.x
        );
        __syncthreads();
        const RoughBergomiHybridGridView grid = {
            far_weights,
            log_variance_corrections,
        };

        double sum = 0.0;
        double sumsq = 0.0;
        for (std::size_t path = threadIdx.x;
             path < monte_carlo_paths_per_price;
             path += blockDim.x) {
            const float payoff = evaluate_path<Side>(
                prepared, grid, history, path
            );
            const double value = static_cast<double>(payoff);
            sum += value;
            sumsq += value * value;
        }

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
        __syncthreads();
    }
}

void validate_rough_bergomi_european_option_launch(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    const float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    const float* device_prices,
    const float* device_standard_errors
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(device_products, "device_products");
    validate_device_pointer(
        device_history_workspace, "device_history_workspace"
    );
    validate_device_pointer(device_prices, "device_prices");
    validate_device_pointer(device_standard_errors, "device_standard_errors");
    validate_model_product_construction(
        model_count, product_count, cartesian_product, result_count
    );
    if (result_offset >= result_count
        || launch_result_count == 0U
        || launch_result_count > result_count - result_offset) {
        throw std::invalid_argument(
            "The rough-Bergomi European-option launch batch exceeds the "
            "result array."
        );
    }
    validate_monte_carlo_parameters(
        monte_carlo_paths_per_price, target_dt
    );
    validate_day_fraction(day_fraction);
    validate_reduction_block_size(threads_per_block);
    validate_block_count(launch_result_count, block_count);
    validate_grid_x_size(block_count);
    validate_row_seed_range(result_count, base_seed);
    if (maximum_step_count == 0U
        || maximum_step_count
            > static_cast<std::size_t>(
                std::numeric_limits<std::uint32_t>::max()
            )) {
        throw std::invalid_argument(
            "rough-Bergomi maximum_step_count must fit uint32_t."
        );
    }
    const std::size_t required_history = checked_product(
        checked_product(
            block_count,
            threads_per_block,
            "rough-Bergomi history lane count overflows size_t."
        ),
        maximum_step_count,
        "rough-Bergomi history element count overflows size_t."
    );
    if (history_workspace_float_count < required_history) {
        throw std::invalid_argument(
            "The rough-Bergomi history workspace is smaller than the "
            "configured persistent grid."
        );
    }
}

}  // namespace

RoughBergomiWorkspacePlan plan_european_option_workspace(
    const product::EuropeanOptionParameters* host_products,
    std::size_t product_count,
    float day_fraction,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count
) {
    if (host_products == nullptr || product_count == 0U) {
        throw std::invalid_argument(
            "rough-Bergomi workspace planning requires host product rows."
        );
    }
    if (!std::isfinite(target_dt) || !(target_dt > 0.0f)) {
        throw std::invalid_argument(
            "rough-Bergomi workspace target_dt must be finite and positive."
        );
    }
    validate_day_fraction(day_fraction);
    validate_reduction_block_size(threads_per_block);
    if (block_count == 0U) {
        throw std::invalid_argument(
            "rough-Bergomi workspace block_count must be positive."
        );
    }

    std::size_t maximum_step_count = 1U;
    for (std::size_t product_index = 0U;
         product_index < product_count;
         ++product_index) {
        const std::uint32_t maturity_days =
            host_products[product_index].maturity;
        if (maturity_days == 0U) {
            throw std::invalid_argument(
                "rough-Bergomi workspace maturities must be positive."
            );
        }
        const float maturity =
            static_cast<float>(maturity_days) * day_fraction;
        maximum_step_count = std::max(
            maximum_step_count,
            rounded_step_count(maturity, target_dt)
        );
    }
    if (maximum_step_count
        > static_cast<std::size_t>(
            std::numeric_limits<std::uint32_t>::max()
        )) {
        throw std::overflow_error(
            "rough-Bergomi maximum step count exceeds uint32_t."
        );
    }
    const std::size_t history_float_count = checked_product(
        checked_product(
            block_count,
            threads_per_block,
            "rough-Bergomi planned history lane count overflows size_t."
        ),
        maximum_step_count,
        "rough-Bergomi planned history size overflows size_t."
    );
    const std::size_t history_bytes = checked_product(
        history_float_count,
        sizeof(float),
        "rough-Bergomi planned history byte count overflows size_t."
    );
    const std::size_t dynamic_shared_bytes =
        reduction_shared_bytes(threads_per_block)
        + hybrid_grid_shared_bytes(maximum_step_count);
    return {
        maximum_step_count,
        history_float_count,
        history_bytes,
        dynamic_shared_bytes,
    };
}

template<OptionSide Side>
void launch_rough_bergomi_european_option_cuda(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    validate_rough_bergomi_european_option_launch(
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
        target_dt,
        threads_per_block,
        block_count,
        maximum_step_count,
        device_history_workspace,
        history_workspace_float_count,
        base_seed,
        device_prices,
        device_standard_errors
    );

    const std::size_t shared_bytes =
        reduction_shared_bytes(threads_per_block)
        + hybrid_grid_shared_bytes(maximum_step_count);
    int active_blocks_per_multiprocessor = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_multiprocessor,
            rough_bergomi_european_option_kernel<Side>,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        "rough-Bergomi European option occupancy check"
    );
    if (active_blocks_per_multiprocessor == 0) {
        throw std::invalid_argument(
            "rough-Bergomi European-option kernel cannot reside on an SM."
        );
    }

    report_cuda_kernel_launch_if_enabled(
        "rough_bergomi.european_option",
        option_side_name(Side),
        rough_bergomi_european_option_kernel<Side>,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    rough_bergomi_european_option_kernel<Side><<<
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
        target_dt,
        maximum_step_count,
        device_history_workspace,
        base_seed,
        device_prices,
        device_standard_errors
    );
    check_cuda(cudaGetLastError(), "rough-Bergomi European option kernel");
}

using LaunchSignature = decltype(
    launch_rough_bergomi_european_option_cuda<OptionSide::call>
);
namespace {
[[maybe_unused]] LaunchSignature* launch_instantiation_0 =
    &launch_rough_bergomi_european_option_cuda<OptionSide::call>;
}  // namespace
namespace {
[[maybe_unused]] LaunchSignature* launch_instantiation_1 =
    &launch_rough_bergomi_european_option_cuda<OptionSide::put>;
}  // namespace

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
