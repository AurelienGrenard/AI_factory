// Persistent exact CIR terminal and regular-calendar sample kernels.
#include "model/fixed_income/cir/sample.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/sample.cuh"
#include "common/simulation/path_simulation.cuh"

#include "model/fixed_income/cir/dynamics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::cir {
namespace {

struct StateObservationWriter {
    float* states;
    std::size_t observation_stride;

    __device__ __forceinline__ bool on_initial_state(float) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        float state
    ) {
        states[static_cast<std::size_t>(observation) * observation_stride] =
            state;
        return true;
    }
};

__global__ void cir_terminal_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    float maturity,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ states
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
        const DynamicsPolicy::PreparedModel prepared_model =
            DynamicsPolicy::prepare_model(model);
        const DynamicsPolicy::PreparedTransition transition =
            DynamicsPolicy::prepare_transition(prepared_model, maturity);
        states[sample_index] =
            simulation::simulate_exact_transition_terminal<DynamicsPolicy>(
                prepared_model,
                transition,
                philox::make_key(base_seed + indices.model_index),
                indices.path_index
            );
    }
}

__global__ void cir_calendar_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    std::size_t total_sample_count,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ states
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
        const DynamicsPolicy::PreparedModel prepared_model =
            DynamicsPolicy::prepare_model(model);
        const DynamicsPolicy::PreparedTransition initial_stub_transition =
            DynamicsPolicy::prepare_transition(
                prepared_model, first_observation_time
            );
        const DynamicsPolicy::PreparedTransition regular_transition =
            DynamicsPolicy::prepare_transition(
                prepared_model, observation_interval
            );
        StateObservationWriter writer{states + sample_index, total_sample_count};
        simulation::simulate_exact_transition_stubbed_regular_schedule<
            DynamicsPolicy
        >(
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

std::size_t validate_cir_sample_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    const float* device_states
) {
    return sample::validate_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_states
    );
}

}  // namespace

void launch_cir_terminal_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float maturity,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_states
) {
    validate_cir_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_states
    );
    sample::validate_terminal_time(maturity);
    report_cuda_kernel_launch_if_enabled(
        "cir.samples", "terminal", cir_terminal_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    cir_terminal_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, maturity, sample_offset,
        launch_sample_count, base_seed, device_states
    );
    check_cuda(cudaGetLastError(), "CIR terminal sample kernel");
}

void launch_cir_calendar_samples_cuda(
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
    float* device_states
) {
    const std::size_t total_sample_count = validate_cir_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_states
    );
    sample::validate_regular_calendar(
        first_observation_time, observation_interval, observation_count
    );
    report_cuda_kernel_launch_if_enabled(
        "cir.samples", "calendar", cir_calendar_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    cir_calendar_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, total_sample_count,
        first_observation_time, observation_interval, observation_count,
        sample_offset, launch_sample_count, base_seed, device_states
    );
    check_cuda(cudaGetLastError(), "CIR calendar sample kernel");
}

}  // namespace ai_factory::workbench::model::fixed_income::cir
