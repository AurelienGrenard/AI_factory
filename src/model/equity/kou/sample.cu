// Persistent kou terminal and regular-calendar sample kernels.
#include "model/equity/kou/sample.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/sample.cuh"

// Include the reusable dynamics so NVCC can inline every transition.
#include "model/equity/kou/dynamics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::kou {
namespace {

__global__ void kou_terminal_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    float maturity,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ spots
) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t launch_index = thread;
         launch_index < launch_sample_count;
         launch_index += stride) {
        const std::size_t sample_index = sample_offset + launch_index;
        const sample::ModelPathIndices indices =
            sample::decode_sample_index(sample_index, paths_per_model);
        const PreparedModel prepared =
            prepare_model(models[indices.model_index]);
        const PreparedTransition transition =
            prepare_transition(prepared, maturity);
        const State terminal = simulate_terminal_state(
            prepared,
            transition,
            philox::make_key(base_seed + indices.model_index),
            indices.path_index
        );
        spots[sample_index] = expf(terminal.log_spot);
    }
}

__global__ void kou_calendar_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    std::size_t total_sample_count,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ spots
) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t launch_index = thread;
         launch_index < launch_sample_count;
         launch_index += stride) {
        const std::size_t sample_index = sample_offset + launch_index;
        const sample::ModelPathIndices indices =
            sample::decode_sample_index(sample_index, paths_per_model);
        const ModelParameters model = models[indices.model_index];
        const PreparedModel prepared_model = prepare_model(model);
        const PreparedTransition initial_stub_transition =
            prepare_transition(prepared_model, first_observation_time);
        const PreparedTransition regular_transition =
            prepare_transition(prepared_model, observation_interval);
        const State terminal = simulate_on_regular_grid(
            prepared_model,
            initial_stub_transition,
            regular_transition,
            philox::make_key(base_seed + indices.model_index),
            indices.path_index,
            observation_count,
            total_sample_count,
            spots + sample_index
        );
        spots[(static_cast<std::size_t>(observation_count) - 1U)
                  * total_sample_count + sample_index] =
            expf(terminal.log_spot);
    }
}

std::size_t validate_kou_sample_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    const float* device_spots
) {
    return sample::validate_sample_launch(
        device_models,
        model_count,
        paths_per_model,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        base_seed,
        device_spots
    );
}

}  // namespace

void launch_kou_terminal_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float maturity,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_spots
) {
    validate_kou_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_spots
    );
    sample::validate_terminal_time(maturity);
    report_cuda_kernel_launch_if_enabled(
        "kou.samples",
        "terminal",
        kou_terminal_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    kou_terminal_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, maturity, sample_offset,
        launch_sample_count, base_seed, device_spots
    );
    check_cuda(cudaGetLastError(), "kou terminal sample kernel");
}

void launch_kou_calendar_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_spots
) {
    const std::size_t total_sample_count =
        validate_kou_sample_launch(
            device_models, model_count, paths_per_model, sample_offset,
            launch_sample_count, threads_per_block, block_count, base_seed,
            device_spots
        );
    sample::validate_regular_calendar(
        first_observation_time, observation_interval, observation_count
    );
    report_cuda_kernel_launch_if_enabled(
        "kou.samples",
        "calendar",
        kou_calendar_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    kou_calendar_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, total_sample_count,
        first_observation_time, observation_interval, observation_count,
        sample_offset, launch_sample_count, base_seed, device_spots
    );
    check_cuda(cudaGetLastError(), "kou calendar sample kernel");
}

}  // namespace ai_factory::workbench::kou
