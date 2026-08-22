// Persistent variance gamma terminal and regular-calendar sample kernels.
#include "model/equity/variance_gamma/sample.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/equity/observation_handlers.cuh"
#include "common/equity/path_simulation.cuh"
#include "common/sample.cuh"

// Include the reusable dynamics so NVCC can inline every transition.
#include "model/equity/variance_gamma/dynamics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::variance_gamma {
namespace {

__global__ void variance_gamma_terminal_samples_kernel(
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
        const State terminal = equity::simulate_exact_transition_terminal<
            DynamicsPolicy
        >(
            prepared,
            transition,
            philox::make_key(base_seed + indices.model_index),
            indices.path_index
        );
        spots[sample_index] = expf(terminal.log_spot);
    }
}

__global__ void variance_gamma_calendar_samples_kernel(
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
        equity::SpotObservationWriter<DynamicsPolicy> writer{
            spots + sample_index,
            total_sample_count,
            observation_count,
        };
        equity::simulate_exact_transition_regular_schedule<DynamicsPolicy>(
            prepared_model,
            initial_stub_transition,
            regular_transition,
            observation_count,
            philox::make_key(base_seed + indices.model_index),
            indices.path_index,
            writer
        );
    }
}

std::size_t validate_variance_gamma_sample_launch(
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

void launch_variance_gamma_terminal_samples_cuda(
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
    validate_variance_gamma_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_spots
    );
    sample::validate_terminal_time(maturity);
    report_cuda_kernel_launch_if_enabled(
        "variance_gamma.samples",
        "terminal",
        variance_gamma_terminal_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    variance_gamma_terminal_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, maturity, sample_offset,
        launch_sample_count, base_seed, device_spots
    );
    check_cuda(cudaGetLastError(), "variance gamma terminal sample kernel");
}

void launch_variance_gamma_calendar_samples_cuda(
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
        validate_variance_gamma_sample_launch(
            device_models, model_count, paths_per_model, sample_offset,
            launch_sample_count, threads_per_block, block_count, base_seed,
            device_spots
        );
    sample::validate_regular_calendar(
        first_observation_time, observation_interval, observation_count
    );
    report_cuda_kernel_launch_if_enabled(
        "variance_gamma.samples",
        "calendar",
        variance_gamma_calendar_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    variance_gamma_calendar_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, total_sample_count,
        first_observation_time, observation_interval, observation_count,
        sample_offset, launch_sample_count, base_seed, device_spots
    );
    check_cuda(cudaGetLastError(), "variance gamma calendar sample kernel");
}

}  // namespace ai_factory::workbench::variance_gamma
