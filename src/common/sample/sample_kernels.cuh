// Generic persistent CUDA kernels for model-only state sampling.
#pragma once

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/sample/concepts.cuh"
#include "common/sample/sources.cuh"
#include "common/simulation/schedule.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::sample {

inline constexpr std::size_t kMaximumSharedPreparedInputBytes = 2048U;

struct SampleIndices {
    std::size_t parameter_index;
    std::size_t path_index;
};

__host__ __device__ constexpr SampleIndices decode_sample(
    std::size_t sample_index,
    std::size_t paths_per_parameter
) noexcept {
    return {
        sample_index / paths_per_parameter,
        sample_index % paths_per_parameter,
    };
}

inline std::size_t total_sample_count(const SampleRange& range) {
    if (range.parameter_count == 0U
        || range.paths_per_parameter == 0U) {
        throw std::invalid_argument(
            "Sampling requires positive parameter and path counts."
        );
    }
    if (range.parameter_count
        > std::numeric_limits<std::size_t>::max()
            / range.paths_per_parameter) {
        throw std::overflow_error(
            "The total sample count exceeds size_t."
        );
    }
    return range.parameter_count * range.paths_per_parameter;
}

inline std::size_t launch_parameter_count(
    const SampleRange& range
) {
    const std::size_t first =
        range.sample_offset / range.paths_per_parameter;
    const std::size_t final =
        (range.sample_offset + range.launch_sample_count - 1U)
            / range.paths_per_parameter;
    return final - first + 1U;
}

template<ExecutableSamplingPolicy Policy>
__device__ __forceinline__ void simulate_and_write_sample(
    const typename Policy::Schedule::PreparedSchedule& prepared,
    std::size_t sample_index,
    SampleIndices indices,
    std::size_t total_samples,
    SamplingSeeds seeds,
    const typename Policy::Output& output
) {
    using Schedule = typename Policy::Schedule;
    using Observation = typename Policy::Observation;
    const philox::PhiloxKey dynamics_key = philox::make_key(
        seeds.dynamics + indices.parameter_index
    );

    if constexpr (simulation::TerminalSchedulePolicy<Schedule>) {
        const typename Schedule::Dynamics::State terminal =
            Schedule::simulate_terminal(
                prepared,
                dynamics_key,
                indices.path_index
            );
        Observation::write_terminal(output, sample_index, terminal);
    } else {
        typename Observation::Handler handler = Observation::make_handler(
            output,
            sample_index,
            total_samples
        );
        Schedule::simulate(
            prepared,
            dynamics_key,
            indices.path_index,
            handler
        );
    }
}

template<
    ExecutableSamplingPolicy Policy,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
__device__ __forceinline__ void evaluate_sample_from_prepared_input(
    const typename Policy::Schedule::PreparedInput& prepared_input,
    const CalendarSource& calendar_source,
    const typename Policy::Schedule::TimeConfiguration& time_configuration,
    std::size_t sample_index,
    SampleIndices indices,
    std::size_t total_samples,
    SamplingSeeds seeds,
    const typename Policy::Output& output
) {
    using Schedule = typename Policy::Schedule;
    const typename Schedule::Calendar calendar = calendar_source.load(
        sample_index,
        seeds.schedule
    );
    const typename Schedule::PreparedSchedule prepared =
        Schedule::prepare_from_input(
            prepared_input,
            calendar,
            time_configuration
        );
    simulate_and_write_sample<Policy>(
        prepared,
        sample_index,
        indices,
        total_samples,
        seeds,
        output
    );
}

template<
    SamplingPolicy Policy,
    ParameterSourceFor<typename Policy::Parameters> ParameterSource
>
struct DevicePreparedInputAdapter {
    ParameterSource parameters;
    typename Policy::Schedule::TimeConfiguration time_configuration;

    void validate(std::size_t parameter_count) const {
        parameters.validate(parameter_count);
    }

    __device__ __forceinline__ typename Policy::Schedule::PreparedInput load(
        std::size_t parameter_index,
        std::uint64_t seed
    ) const {
        return Policy::Schedule::prepare_input(
            parameters.load(parameter_index, seed),
            time_configuration
        );
    }
};

template<
    ExecutableSamplingPolicy Policy,
    ParameterSourceFor<
        typename Policy::Schedule::PreparedInput
    > PreparedInputSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
__global__ void thread_grid_stride_sample_kernel(
    PreparedInputSource prepared_input_source,
    CalendarSource calendar_source,
    SampleRange range,
    std::size_t total_samples,
    typename Policy::Schedule::TimeConfiguration time_configuration,
    SamplingSeeds seeds,
    typename Policy::Output output
) {
    const std::size_t thread =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride =
        static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t launch_index = thread;
         launch_index < range.launch_sample_count;
         launch_index += stride) {
        const std::size_t sample_index = range.sample_offset + launch_index;
        const SampleIndices indices = decode_sample(
            sample_index,
            range.paths_per_parameter
        );
        const typename Policy::Schedule::PreparedInput prepared_input =
            prepared_input_source.load(
                indices.parameter_index,
                seeds.parameters
            );
        evaluate_sample_from_prepared_input<Policy>(
            prepared_input,
            calendar_source,
            time_configuration,
            sample_index,
            indices,
            total_samples,
            seeds,
            output
        );
    }
}

template<
    ExecutableSamplingPolicy Policy,
    ParameterSourceFor<
        typename Policy::Schedule::PreparedInput
    > PreparedInputSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
__global__ void parameter_block_sample_kernel(
    PreparedInputSource prepared_input_source,
    CalendarSource calendar_source,
    SampleRange range,
    std::size_t total_samples,
    typename Policy::Schedule::TimeConfiguration time_configuration,
    SamplingSeeds seeds,
    typename Policy::Output output
) {
    static_assert(
        sizeof(typename Policy::Schedule::PreparedInput)
            <= kMaximumSharedPreparedInputBytes,
        "Prepared model input exceeds the 2048-byte per-block shared-memory "
        "budget; store a compact view and keep large coefficients in a "
        "device-resident pool."
    );
    __shared__ typename Policy::Schedule::PreparedInput prepared_input;

    const std::size_t first_parameter =
        range.sample_offset / range.paths_per_parameter;
    const std::size_t final_sample =
        range.sample_offset + range.launch_sample_count - 1U;
    const std::size_t final_parameter =
        final_sample / range.paths_per_parameter;

    for (std::size_t parameter_index = first_parameter + blockIdx.x;
         parameter_index <= final_parameter;
         parameter_index += gridDim.x) {
        if (threadIdx.x == 0U) {
            prepared_input = prepared_input_source.load(
                parameter_index,
                seeds.parameters
            );
        }
        __syncthreads();

        for (std::size_t path_index = threadIdx.x;
             path_index < range.paths_per_parameter;
             path_index += blockDim.x) {
            const std::size_t sample_index =
                parameter_index * range.paths_per_parameter + path_index;
            if (sample_index >= range.sample_offset
                && sample_index <= final_sample) {
                evaluate_sample_from_prepared_input<Policy>(
                    prepared_input,
                    calendar_source,
                    time_configuration,
                    sample_index,
                    {parameter_index, path_index},
                    total_samples,
                    seeds,
                    output
                );
            }
        }
        __syncthreads();
    }
}

inline SampleExecutionStrategy resolve_execution_strategy(
    SampleExecutionStrategy strategy,
    std::size_t paths_per_parameter
) {
    if (strategy == SampleExecutionStrategy::automatic) {
        return paths_per_parameter == 1U
            ? SampleExecutionStrategy::thread_grid_stride
            : SampleExecutionStrategy::parameter_block;
    }
    return strategy;
}

template<
    ExecutableSamplingPolicy Policy,
    typename InputSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
inline void validate_sample_launch_common(
    const InputSource& input_source,
    const CalendarSource& calendar_source,
    const SampleRange& range,
    const typename Policy::Schedule::TimeConfiguration& time_configuration,
    const SampleLaunchConfiguration& launch_configuration,
    SamplingSeeds seeds,
    const typename Policy::Output& output,
    SampleExecutionStrategy strategy
) {
    const std::size_t total_samples = total_sample_count(range);
    if (range.sample_offset >= total_samples
        || range.launch_sample_count == 0U
        || range.launch_sample_count
            > total_samples - range.sample_offset) {
        throw std::invalid_argument(
            "The sample launch batch exceeds the logical sample array."
        );
    }

    input_source.validate(range.parameter_count);
    calendar_source.validate(total_samples);
    Policy::Observation::validate(output);
    simulation::validate_time_configuration(time_configuration);
    validate_cuda_block_size(launch_configuration.threads_per_block);
    validate_grid_x_size(launch_configuration.block_count);
    validate_row_seed_range(range.parameter_count, seeds.dynamics);
    validate_block_count(
        strategy == SampleExecutionStrategy::parameter_block
            ? launch_parameter_count(range)
            : range.launch_sample_count,
        launch_configuration.block_count
    );
}

template<
    SamplingPolicy Policy,
    ParameterSourceFor<typename Policy::Parameters> ParameterSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
inline void validate_sample_launch(
    const ParameterSource& parameter_source,
    const CalendarSource& calendar_source,
    const SampleRange& range,
    const typename Policy::Schedule::TimeConfiguration& time_configuration,
    const SampleLaunchConfiguration& launch_configuration,
    SamplingSeeds seeds,
    const typename Policy::Output& output,
    SampleExecutionStrategy strategy
) {
    validate_sample_launch_common<Policy>(
        parameter_source,
        calendar_source,
        range,
        time_configuration,
        launch_configuration,
        seeds,
        output,
        strategy
    );
}

template<
    SamplingPolicy Policy,
    ParameterSourceFor<typename Policy::Parameters> ParameterSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
inline void launch_samples_cuda(
    const ParameterSource& parameter_source,
    const CalendarSource& calendar_source,
    const SampleRange& range,
    const typename Policy::Schedule::TimeConfiguration& time_configuration,
    const SampleLaunchConfiguration& launch_configuration,
    SamplingSeeds seeds,
    const typename Policy::Output& output,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name,
    SampleExecutionStrategy requested_strategy =
        SampleExecutionStrategy::automatic
) {
    using PreparedInputSource = DevicePreparedInputAdapter<
        Policy,
        ParameterSource
    >;
    const PreparedInputSource prepared_input_source{
        parameter_source,
        time_configuration,
    };
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        requested_strategy,
        range.paths_per_parameter
    );
    validate_sample_launch<Policy>(
        parameter_source,
        calendar_source,
        range,
        time_configuration,
        launch_configuration,
        seeds,
        output,
        strategy
    );

    const dim3 grid(
        static_cast<unsigned int>(launch_configuration.block_count)
    );
    const dim3 block(launch_configuration.threads_per_block);
    const std::size_t total_samples = total_sample_count(range);

    if (strategy == SampleExecutionStrategy::parameter_block) {
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            parameter_block_sample_kernel<
                Policy,
                PreparedInputSource,
                CalendarSource
            >,
            grid,
            block
        );
        parameter_block_sample_kernel<
            Policy,
            PreparedInputSource,
            CalendarSource
        ><<<grid, block>>>(
            prepared_input_source,
            calendar_source,
            range,
            total_samples,
            time_configuration,
            seeds,
            output
        );
    } else {
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            thread_grid_stride_sample_kernel<
                Policy,
                PreparedInputSource,
                CalendarSource
            >,
            grid,
            block
        );
        thread_grid_stride_sample_kernel<
            Policy,
            PreparedInputSource,
            CalendarSource
        ><<<grid, block>>>(
            prepared_input_source,
            calendar_source,
            range,
            total_samples,
            time_configuration,
            seeds,
            output
        );
    }
    check_cuda(cudaGetLastError(), operation_name);
}

template<
    ExternallyPreparedSamplingPolicy Policy,
    ParameterSourceFor<
        typename Policy::Schedule::PreparedInput
    > PreparedInputSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
inline void launch_prepared_samples_cuda(
    const PreparedInputSource& prepared_input_source,
    const CalendarSource& calendar_source,
    const SampleRange& range,
    const typename Policy::Schedule::TimeConfiguration& time_configuration,
    const SampleLaunchConfiguration& launch_configuration,
    SamplingSeeds seeds,
    const typename Policy::Output& output,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name,
    SampleExecutionStrategy requested_strategy =
        SampleExecutionStrategy::automatic
) {
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        requested_strategy,
        range.paths_per_parameter
    );
    validate_sample_launch_common<Policy>(
        prepared_input_source,
        calendar_source,
        range,
        time_configuration,
        launch_configuration,
        seeds,
        output,
        strategy
    );

    const std::size_t total_samples = total_sample_count(range);
    const dim3 grid(
        static_cast<unsigned int>(launch_configuration.block_count)
    );
    const dim3 block(launch_configuration.threads_per_block);
    if (strategy == SampleExecutionStrategy::parameter_block) {
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            parameter_block_sample_kernel<
                Policy,
                PreparedInputSource,
                CalendarSource
            >,
            grid,
            block
        );
        parameter_block_sample_kernel<
            Policy,
            PreparedInputSource,
            CalendarSource
        ><<<grid, block>>>(
            prepared_input_source,
            calendar_source,
            range,
            total_samples,
            time_configuration,
            seeds,
            output
        );
    } else {
        report_cuda_kernel_launch_if_enabled(
            diagnostic_name,
            diagnostic_variant,
            thread_grid_stride_sample_kernel<
                Policy,
                PreparedInputSource,
                CalendarSource
            >,
            grid,
            block
        );
        thread_grid_stride_sample_kernel<
            Policy,
            PreparedInputSource,
            CalendarSource
        ><<<grid, block>>>(
            prepared_input_source,
            calendar_source,
            range,
            total_samples,
            time_configuration,
            seeds,
            output
        );
    }
    check_cuda(cudaGetLastError(), operation_name);
}

}  // namespace ai_factory::workbench::sample
