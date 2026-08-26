// Persistent heston terminal and regular-calendar sample kernels.
#include "model/equity/heston/sample.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/equity/handlers.cuh"
#include "common/simulation/path_simulation.cuh"
#include "common/sample.cuh"

// Include the reusable dynamics so NVCC can inline every transition.
#include "model/equity/heston/dynamics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::heston {
namespace {

__global__ void heston_terminal_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    float dt,
    std::uint32_t step_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ spots,
    float* __restrict__ variances
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
        const PreparedModel prepared = prepare_model(models[indices.model_index], dt);
        const State terminal = simulation::simulate_fixed_step_terminal<
            DynamicsPolicy
        >(
            prepared,
            step_count,
            philox::make_key(base_seed + indices.model_index),
            indices.path_index
        );
        spots[sample_index] = expf(terminal.log_spot);
        variances[sample_index] = terminal.variance;
    }
}

__global__ void heston_calendar_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    std::size_t total_sample_count,
    float dt,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ spots,
    float* __restrict__ variances
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
        const PreparedModel prepared_model = prepare_model(model, dt);
        ::ai_factory::workbench::equity::SpotAndStateObservationWriter<
            DynamicsPolicy,
            &State::variance
        > writer{
            spots + sample_index,
            variances + sample_index,
            total_sample_count,
            observation_count,
        };
        simulation::simulate_fixed_step_stubbed_regular_schedule<DynamicsPolicy>(
            prepared_model,
            initial_stub_steps,
            steps_per_observation,
            observation_count,
            philox::make_key(base_seed + indices.model_index),
            indices.path_index,
            writer
        );
    }
}

std::size_t validate_heston_sample_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    const float* device_spots,
    const float* device_variances
) {
    const std::size_t total_sample_count = sample::validate_sample_launch(
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
    validate_device_pointer(device_variances, "device_variances");
    return total_sample_count;
}

}  // namespace

void launch_heston_terminal_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float maturity,
    float dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_spots,
    float* device_variances
) {
    validate_heston_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_spots, device_variances
    );
    sample::validate_terminal_time(maturity);
    const std::uint32_t step_count =
        sample::rounded_step_count(maturity, dt);
    report_cuda_kernel_launch_if_enabled(
        "heston.samples",
        "terminal",
        heston_terminal_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    heston_terminal_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, dt, step_count, sample_offset,
        launch_sample_count, base_seed, device_spots, device_variances
    );
    check_cuda(cudaGetLastError(), "heston terminal sample kernel");
}

void launch_heston_calendar_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    float dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_spots,
    float* device_variances
) {
    const std::size_t total_sample_count =
        validate_heston_sample_launch(
            device_models, model_count, paths_per_model, sample_offset,
            launch_sample_count, threads_per_block, block_count, base_seed,
            device_spots, device_variances
        );
    sample::validate_regular_calendar(
        first_observation_time, observation_interval, observation_count
    );
    const std::uint32_t initial_stub_steps =
        sample::rounded_step_count(first_observation_time, dt);
    const std::uint32_t steps_per_observation =
        sample::rounded_step_count(observation_interval, dt);
    sample::calendar_step_count(
        initial_stub_steps, steps_per_observation, observation_count
    );
    report_cuda_kernel_launch_if_enabled(
        "heston.samples",
        "calendar",
        heston_calendar_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    heston_calendar_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, total_sample_count,
        dt, initial_stub_steps,
        steps_per_observation, observation_count, sample_offset,
        launch_sample_count, base_seed, device_spots, device_variances
    );
    check_cuda(cudaGetLastError(), "heston calendar sample kernel");
}

}  // namespace ai_factory::workbench::model::equity::heston
