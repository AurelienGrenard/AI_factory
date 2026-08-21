// Persistent exact centered-factor samples for G2++.
#include "model/fixed_income/g2_plus_plus/sample.cuh"

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/sample.cuh"

#include "model/fixed_income/g2/dynamics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::g2_plus_plus {
namespace {

using model::g2::PreparedTransition;
using model::g2::State;

__global__ void g2_plus_plus_terminal_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    float maturity,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ states_x,
    float* __restrict__ states_y
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
        const model::g2::PreparedModel prepared_model =
            model::g2::prepare_model(models[indices.model_index].process);
        const PreparedTransition transition =
            model::g2::prepare_transition(prepared_model, maturity);
        const State terminal = model::g2::simulate_terminal_state(
            prepared_model,
            transition,
            {0.0f, 0.0f},
            philox::make_key(base_seed + indices.model_index),
            indices.path_index
        );
        states_x[sample_index] = terminal.state_x;
        states_y[sample_index] = terminal.state_y;
    }
}

__global__ void g2_plus_plus_calendar_samples_kernel(
    const ModelParameters* __restrict__ models,
    std::size_t paths_per_model,
    std::size_t total_sample_count,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::uint64_t base_seed,
    float* __restrict__ states_x,
    float* __restrict__ states_y
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
        const auto process = models[indices.model_index].process;
        const model::g2::PreparedModel prepared_model =
            model::g2::prepare_model(process);
        const PreparedTransition initial_stub_transition =
            model::g2::prepare_transition(
                prepared_model, first_observation_time
            );
        const PreparedTransition regular_transition =
            model::g2::prepare_transition(
                prepared_model, observation_interval
            );
        const State terminal = model::g2::simulate_on_regular_grid(
            prepared_model,
            initial_stub_transition,
            regular_transition,
            {0.0f, 0.0f},
            philox::make_key(base_seed + indices.model_index),
            indices.path_index,
            observation_count,
            total_sample_count,
            states_x + sample_index,
            states_y + sample_index
        );
        const std::size_t terminal_index =
            (static_cast<std::size_t>(observation_count) - 1U)
                * total_sample_count + sample_index;
        states_x[terminal_index] = terminal.state_x;
        states_y[terminal_index] = terminal.state_y;
    }
}

std::size_t validate_g2_plus_plus_sample_launch(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    const float* device_states_x,
    const float* device_states_y
) {
    const std::size_t total = sample::validate_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_states_x
    );
    validate_device_pointer(device_states_y, "device_states_y");
    return total;
}

}  // namespace

void launch_g2_plus_plus_terminal_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float maturity,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_states_x,
    float* device_states_y
) {
    validate_g2_plus_plus_sample_launch(
        device_models, model_count, paths_per_model, sample_offset,
        launch_sample_count, threads_per_block, block_count, base_seed,
        device_states_x, device_states_y
    );
    sample::validate_terminal_time(maturity);
    report_cuda_kernel_launch_if_enabled(
        "g2_plus_plus.samples", "terminal",
        g2_plus_plus_terminal_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    g2_plus_plus_terminal_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, maturity, sample_offset,
        launch_sample_count, base_seed, device_states_x, device_states_y
    );
    check_cuda(cudaGetLastError(), "G2++ terminal sample kernel");
}

void launch_g2_plus_plus_calendar_samples_cuda(
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
    float* device_states_x,
    float* device_states_y
) {
    const std::size_t total_sample_count =
        validate_g2_plus_plus_sample_launch(
            device_models, model_count, paths_per_model, sample_offset,
            launch_sample_count, threads_per_block, block_count, base_seed,
            device_states_x, device_states_y
        );
    sample::validate_regular_calendar(
        first_observation_time, observation_interval, observation_count
    );
    report_cuda_kernel_launch_if_enabled(
        "g2_plus_plus.samples", "calendar",
        g2_plus_plus_calendar_samples_kernel,
        dim3(static_cast<unsigned int>(block_count)),
        dim3(threads_per_block)
    );
    g2_plus_plus_calendar_samples_kernel<<<
        static_cast<unsigned int>(block_count), threads_per_block
    >>>(
        device_models, paths_per_model, total_sample_count,
        first_observation_time, observation_interval, observation_count,
        sample_offset, launch_sample_count, base_seed,
        device_states_x, device_states_y
    );
    check_cuda(cudaGetLastError(), "G2++ calendar sample kernel");
}

}  // namespace ai_factory::workbench::model::g2_plus_plus
