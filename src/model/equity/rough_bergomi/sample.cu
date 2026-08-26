// Block-persistent rough-Bergomi terminal and calendar sample kernels.
#include "model/equity/rough_bergomi/sample.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/sample.cuh"

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

struct PreparedSampleModel {
    RoughBergomiPreparedParameters model;
    philox::PhiloxKey key;
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

std::size_t hybrid_grid_shared_bytes(std::size_t maximum_step_count) {
    return checked_product(
        2U * sizeof(float),
        maximum_step_count,
        "rough-Bergomi sample shared-grid byte count overflows size_t."
    );
}

__global__ void rough_bergomi_terminal_samples_kernel(
    const RoughBergomiModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    float maturity,
    std::uint32_t step_count,
    std::size_t first_model_index,
    std::size_t launch_model_count,
    std::size_t maximum_step_count,
    float* __restrict__ history_workspace,
    std::uint64_t base_seed,
    float* __restrict__ spots
) {
    __shared__ PreparedSampleModel prepared;
    extern __shared__ float dynamic_shared[];
    float* const far_weights = dynamic_shared;
    float* const log_variance_corrections =
        far_weights + maximum_step_count;
    float* const block_history = history_workspace
        + static_cast<std::size_t>(blockIdx.x)
            * maximum_step_count * blockDim.x;
    const RoughBergomiHistoryView history = {
        block_history + threadIdx.x,
        blockDim.x,
    };

    for (std::size_t local_model_index = blockIdx.x;
         local_model_index < launch_model_count;
         local_model_index += gridDim.x) {
        const std::size_t model_index =
            first_model_index + local_model_index;
        if (threadIdx.x == 0U) {
            prepared = {
                prepare_model(models[model_index], maturity, step_count),
                philox::make_key(base_seed + model_index),
            };
        }
        __syncthreads();
        prepare_hybrid_grid(
            prepared.model,
            step_count,
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
        for (std::size_t path_index = threadIdx.x;
             path_index < paths_per_model;
             path_index += blockDim.x) {
            const RoughBergomiState terminal = simulate_terminal_state(
                prepared.model,
                grid,
                history,
                prepared.key,
                path_index,
                step_count
            );
            spots[model_index * paths_per_model + path_index] =
                expf(terminal.log_spot);
        }
        __syncthreads();
    }
}

__global__ void rough_bergomi_calendar_samples_kernel(
    const RoughBergomiModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    std::size_t total_sample_count,
    float maturity,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::uint32_t step_count,
    std::size_t first_model_index,
    std::size_t launch_model_count,
    std::size_t maximum_step_count,
    float* __restrict__ history_workspace,
    std::uint64_t base_seed,
    float* __restrict__ spots
) {
    __shared__ PreparedSampleModel prepared;
    extern __shared__ float dynamic_shared[];
    float* const far_weights = dynamic_shared;
    float* const log_variance_corrections =
        far_weights + maximum_step_count;
    float* const block_history = history_workspace
        + static_cast<std::size_t>(blockIdx.x)
            * maximum_step_count * blockDim.x;
    const RoughBergomiHistoryView history = {
        block_history + threadIdx.x,
        blockDim.x,
    };

    for (std::size_t local_model_index = blockIdx.x;
         local_model_index < launch_model_count;
         local_model_index += gridDim.x) {
        const std::size_t model_index =
            first_model_index + local_model_index;
        if (threadIdx.x == 0U) {
            prepared = {
                prepare_model(models[model_index], maturity, step_count),
                philox::make_key(base_seed + model_index),
            };
        }
        __syncthreads();
        prepare_hybrid_grid(
            prepared.model,
            step_count,
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
        for (std::size_t path_index = threadIdx.x;
             path_index < paths_per_model;
             path_index += blockDim.x) {
            const RoughBergomiState terminal = simulate_on_regular_grid(
                prepared.model,
                grid,
                history,
                prepared.key,
                path_index,
                initial_stub_steps,
                steps_per_observation,
                observation_count,
                total_sample_count,
                spots + model_index * paths_per_model
            );
            const std::size_t sample_index =
                model_index * paths_per_model + path_index;
            spots[(static_cast<std::size_t>(observation_count) - 1U)
                      * total_sample_count + sample_index] =
                expf(terminal.log_spot);
        }
        __syncthreads();
    }
}

struct ValidatedRoughSampleLaunch {
    std::size_t total_sample_count;
    std::size_t first_model_index;
    std::size_t launch_model_count;
};

ValidatedRoughSampleLaunch validate_rough_sample_launch(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    const float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    const float* device_spots
) {
    validate_device_pointer(device_models, "device_models");
    validate_device_pointer(
        device_history_workspace, "device_history_workspace"
    );
    validate_device_pointer(device_spots, "device_spots");
    const std::size_t total_sample_count = sample::sample_count(
        model_count, paths_per_model
    );
    if (sample_offset >= total_sample_count
        || launch_sample_count == 0U
        || launch_sample_count > total_sample_count - sample_offset) {
        throw std::invalid_argument(
            "The rough-Bergomi sample batch exceeds the output arrays."
        );
    }
    if (sample_offset % paths_per_model != 0U
        || launch_sample_count % paths_per_model != 0U) {
        throw std::invalid_argument(
            "A rough-Bergomi sample batch must contain complete model rows."
        );
    }
    const std::size_t launch_model_count =
        launch_sample_count / paths_per_model;
    validate_cuda_block_size(threads_per_block);
    validate_block_count(launch_model_count, block_count);
    validate_grid_x_size(block_count);
    validate_row_seed_range(model_count, base_seed);
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
            "rough-Bergomi sample history lanes overflow size_t."
        ),
        maximum_step_count,
        "rough-Bergomi sample history size overflows size_t."
    );
    if (history_workspace_float_count < required_history) {
        throw std::invalid_argument(
            "The rough-Bergomi sample history workspace is too small."
        );
    }
    return {
        total_sample_count,
        sample_offset / paths_per_model,
        launch_model_count,
    };
}

template<class Kernel>
std::size_t validate_residency(
    Kernel kernel,
    unsigned int threads_per_block,
    std::size_t maximum_step_count
) {
    const std::size_t shared_bytes =
        hybrid_grid_shared_bytes(maximum_step_count);
    int active_blocks_per_multiprocessor = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_multiprocessor,
            kernel,
            static_cast<int>(threads_per_block),
            shared_bytes
        ),
        "rough-Bergomi sample occupancy check"
    );
    if (active_blocks_per_multiprocessor == 0) {
        throw std::invalid_argument(
            "The rough-Bergomi sample kernel cannot reside on an SM."
        );
    }
    return shared_bytes;
}

}  // namespace

RoughBergomiSampleWorkspacePlan plan_sample_workspace(
    float maturity,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count
) {
    sample::validate_terminal_time(maturity);
    const std::size_t maximum_step_count =
        sample::rounded_step_count(maturity, target_dt);
    validate_cuda_block_size(threads_per_block);
    if (block_count == 0U) {
        throw std::invalid_argument(
            "rough-Bergomi sample block_count must be positive."
        );
    }
    const std::size_t history_float_count = checked_product(
        checked_product(
            block_count,
            threads_per_block,
            "rough-Bergomi planned sample history lanes overflow size_t."
        ),
        maximum_step_count,
        "rough-Bergomi planned sample history size overflows size_t."
    );
    return {
        maximum_step_count,
        history_float_count,
        checked_product(
            history_float_count,
            sizeof(float),
            "rough-Bergomi planned sample history bytes overflow size_t."
        ),
        hybrid_grid_shared_bytes(maximum_step_count),
    };
}

void launch_rough_bergomi_terminal_samples_cuda(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float maturity,
    float target_dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    float* device_spots
) {
    const ValidatedRoughSampleLaunch launch = validate_rough_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count,
        maximum_step_count, device_history_workspace,
        history_workspace_float_count, base_seed, device_spots
    );
    sample::validate_terminal_time(maturity);
    const std::uint32_t step_count =
        sample::rounded_step_count(maturity, target_dt);
    if (step_count > maximum_step_count) {
        throw std::invalid_argument(
            "rough-Bergomi terminal steps exceed maximum_step_count."
        );
    }
    const std::size_t shared_bytes = validate_residency(
        rough_bergomi_terminal_samples_kernel,
        threads_per_block,
        maximum_step_count
    );
    report_cuda_kernel_launch_if_enabled(
        "rough_bergomi.samples", "terminal",
        rough_bergomi_terminal_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    rough_bergomi_terminal_samples_kernel<<<
        static_cast<unsigned int>(block_count),
        threads_per_block,
        shared_bytes
    >>>(
        device_models, paths_per_model, maturity, step_count,
        launch.first_model_index, launch.launch_model_count,
        maximum_step_count, device_history_workspace, base_seed, device_spots
    );
    check_cuda(cudaGetLastError(), "rough-Bergomi terminal sample kernel");
}

void launch_rough_bergomi_calendar_samples_cuda(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    float target_dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    float* device_spots
) {
    const ValidatedRoughSampleLaunch launch = validate_rough_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count,
        maximum_step_count, device_history_workspace,
        history_workspace_float_count, base_seed, device_spots
    );
    sample::validate_regular_calendar(
        first_observation_time, observation_interval, observation_count
    );
    const std::uint32_t initial_stub_steps =
        sample::rounded_step_count(first_observation_time, target_dt);
    const std::uint32_t steps_per_observation =
        sample::rounded_step_count(observation_interval, target_dt);
    const std::uint32_t step_count = sample::calendar_step_count(
        initial_stub_steps, steps_per_observation, observation_count
    );
    if (step_count > maximum_step_count) {
        throw std::invalid_argument(
            "rough-Bergomi calendar steps exceed maximum_step_count."
        );
    }
    const float maturity = first_observation_time
        + static_cast<float>(observation_count - 1U) * observation_interval;
    const std::size_t shared_bytes = validate_residency(
        rough_bergomi_calendar_samples_kernel,
        threads_per_block,
        maximum_step_count
    );
    report_cuda_kernel_launch_if_enabled(
        "rough_bergomi.samples", "calendar",
        rough_bergomi_calendar_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block),
        shared_bytes
    );
    rough_bergomi_calendar_samples_kernel<<<
        static_cast<unsigned int>(block_count),
        threads_per_block,
        shared_bytes
    >>>(
        device_models, paths_per_model, launch.total_sample_count,
        maturity, initial_stub_steps, steps_per_observation,
        observation_count, step_count, launch.first_model_index,
        launch.launch_model_count, maximum_step_count,
        device_history_workspace, base_seed, device_spots
    );
    check_cuda(cudaGetLastError(), "rough-Bergomi calendar sample kernel");
}

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
