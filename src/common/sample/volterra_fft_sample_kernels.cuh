// Block-cooperative cuFFTDx sampling for Gaussian-Volterra model paths.
#pragma once

#include "common/check_cuda.cuh"
#include "common/cuda_kernel_diagnostics.cuh"
#include "common/sample/concepts.cuh"
#include "common/sample/model_bindings.cuh"
#include "common/sample/sample_kernels.cuh"
#include "common/sample/sources.cuh"
#include "common/volterra/block_fft_convolution.cuh"
#include "common/volterra/hybrid_fft.cuh"
#include "common/volterra/hybrid_schedule.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::sample::volterra_fft {

inline constexpr std::uint32_t kMaximumSampleStepCount = 1008U;

struct LaunchConfiguration {
    std::uint32_t maximum_step_count;
    std::size_t block_count;
};

template<typename Calendar>
__device__ __forceinline__ std::uint64_t maturity_days(
    const Calendar& calendar
) {
    if constexpr (requires { calendar.maturity_days; }) {
        return calendar.maturity_days;
    } else if constexpr (requires {
        Calendar::kObservationCount;
        calendar.interval_days[0U];
    }) {
        std::uint64_t days = 0U;
        #pragma unroll
        for (std::size_t observation = 0U;
             observation < Calendar::kObservationCount;
             ++observation) {
            days += calendar.interval_days[observation];
        }
        return days;
    } else if constexpr (requires {
        calendar.first_observation_day;
        calendar.observation_interval_days;
        calendar.observation_count;
    }) {
        return static_cast<std::uint64_t>(calendar.first_observation_day)
            + static_cast<std::uint64_t>(calendar.observation_count - 1U)
                * calendar.observation_interval_days;
    } else if constexpr (requires {
        calendar.observation_interval_days;
        calendar.observation_count;
    }) {
        return static_cast<std::uint64_t>(
            calendar.observation_interval_days
        ) * calendar.observation_count;
    } else {
        static_assert(
            std::is_same_v<Calendar, void>,
            "A Volterra sample calendar must expose its maturity in days."
        );
    }
}

template<typename KernelPolicy, typename PathPolicy>
struct PreparedParameter {
    typename KernelPolicy::PreparedKernel kernel;
    typename PathPolicy::PreparedModel model;
    philox::PhiloxKey dynamics_key;
    float sqrt_time_step;
};

__device__ __forceinline__ bool contains_sample(
    const SampleRange& range,
    std::size_t sample_index
) {
    return sample_index >= range.sample_offset
        && sample_index
            < range.sample_offset + range.launch_sample_count;
}

template<
    typename Policy,
    ParameterSourceFor<typename Policy::Parameters> ParameterSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource,
    unsigned int Length,
    class Forward,
    class Inverse
>
requires VolterraFftSamplingPolicy<
    Policy,
    volterra::HybridTimeConfiguration
>
__global__ void parameter_block_sample_kernel(
    ParameterSource parameter_source,
    CalendarSource calendar_source,
    SampleRange range,
    std::size_t total_samples,
    std::uint32_t maximum_step_count,
    SamplingSeeds seeds,
    typename Policy::Output output
) {
    using Kernel = typename Policy::Kernel;
    using Path = typename Policy::Path;
    using Schedule = typename Policy::Schedule;
    using Observation = typename Policy::Observation;
    using Row = PreparedParameter<Kernel, Path>;
    using Complex = typename Forward::value_type;
    constexpr unsigned int ffts_per_block = Forward::ffts_per_block;
    constexpr std::size_t spectrum_bytes =
        static_cast<std::size_t>(Length) * sizeof(float2);
    constexpr std::size_t packed_output_bytes =
        static_cast<std::size_t>(Length)
        * ffts_per_block
        * sizeof(float2);
    constexpr std::size_t transform_bytes =
        static_cast<std::size_t>(Forward::shared_memory_size)
            > static_cast<std::size_t>(Inverse::shared_memory_size)
        ? static_cast<std::size_t>(Forward::shared_memory_size)
        : static_cast<std::size_t>(Inverse::shared_memory_size);
    constexpr std::size_t execution_bytes =
        transform_bytes > packed_output_bytes
        ? transform_bytes
        : packed_output_bytes;

    __shared__ Row row;
    extern __shared__ __align__(16) unsigned char shared_storage[];
    auto* const kernel_spectrum = reinterpret_cast<float2*>(shared_storage);
    auto* const execution_storage = reinterpret_cast<Complex*>(
        shared_storage + spectrum_bytes
    );
    auto* const packed_outputs = reinterpret_cast<float2*>(
        shared_storage + spectrum_bytes
    );
    auto* const volterra_variances = reinterpret_cast<float*>(
        shared_storage + spectrum_bytes + execution_bytes
    );

    const std::size_t first_parameter =
        range.sample_offset / range.paths_per_parameter;
    const std::size_t final_sample =
        range.sample_offset + range.launch_sample_count - 1U;
    const std::size_t final_parameter =
        final_sample / range.paths_per_parameter;
    const unsigned int flat_thread =
        threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int threads_per_block = blockDim.x * blockDim.y;

    for (std::size_t parameter_index = first_parameter + blockIdx.x;
         parameter_index <= final_parameter;
         parameter_index += gridDim.x) {
        if (flat_thread == 0U) {
            const typename Policy::Parameters parameters =
                parameter_source.load(parameter_index, seeds.parameters);
            constexpr float time_step = kSampleFixedStepDt;
            row = {
                Kernel::prepare(
                    Path::kernel_parameters(parameters),
                    time_step
                ),
                Path::prepare_model(parameters, time_step),
                philox::make_key(seeds.dynamics + parameter_index),
                sqrtf(time_step),
            };
        }
        __syncthreads();

        for (std::uint32_t step = flat_thread;
             step < maximum_step_count;
             step += threads_per_block) {
            if constexpr (Path::kUsesVolterraVariance) {
                volterra_variances[step] = Kernel::volterra_variance(
                    row.kernel,
                    static_cast<float>(step + 1U) * kSampleFixedStepDt
                );
            } else {
                volterra_variances[step] = 0.0f;
            }
        }

        Complex kernel_data[Forward::storage_size];
        if (Forward::working_group::is_thread_active()) {
            #pragma unroll
            for (unsigned int item = 0U;
                 item < Forward::input_ept;
                 ++item) {
                const unsigned int index =
                    item * Forward::stride + threadIdx.x;
                float weight = 0.0f;
                if (index + 1U < maximum_step_count) {
                    weight = Kernel::far_cell_weight(
                        row.kernel,
                        index + 2U
                    );
                }
                reinterpret_cast<float2*>(kernel_data)[item] = {
                    weight,
                    0.0f,
                };
            }
        }
        Forward().execute(kernel_data, execution_storage);
        if (threadIdx.y == 0U
            && Forward::working_group::is_thread_active()) {
            #pragma unroll
            for (unsigned int item = 0U;
                 item < Forward::output_ept;
                 ++item) {
                const unsigned int index =
                    item * Forward::stride + threadIdx.x;
                if (index < Length) {
                    kernel_spectrum[index] =
                        reinterpret_cast<float2*>(kernel_data)[item];
                }
            }
        }
        __syncthreads();

        const std::size_t parameter_sample_begin =
            parameter_index * range.paths_per_parameter;
        const std::size_t first_path = parameter_index == first_parameter
            ? range.sample_offset - parameter_sample_begin
            : 0U;
        const std::size_t final_path_exclusive =
            parameter_index == final_parameter
            ? final_sample - parameter_sample_begin + 1U
            : range.paths_per_parameter;
        const std::size_t first_pair = first_path / 2U;
        const std::size_t pair_count =
            (final_path_exclusive + 1U) / 2U;

        for (std::size_t pair_base = first_pair;
             pair_base < pair_count;
             pair_base += ffts_per_block) {
            const unsigned int local_fft = threadIdx.y;
            const std::size_t pair = pair_base + local_fft;
            const std::size_t first_pair_path = 2U * pair;

            struct Loader {
                const Row& row;
                SampleRange range;
                std::size_t parameter_sample_begin;
                std::size_t first_pair_path;
                std::uint32_t maximum_step_count;

                __device__ __forceinline__ float2 operator()(
                    unsigned int step
                ) const {
                    float2 increment{0.0f, 0.0f};
                    if (step >= maximum_step_count) return increment;
                    const std::size_t first_sample =
                        parameter_sample_begin + first_pair_path;
                    if (first_pair_path < range.paths_per_parameter
                        && contains_sample(range, first_sample)) {
                        increment.x = row.sqrt_time_step
                            * volterra::hybrid_fft::normal_at(
                                row.dynamics_key,
                                first_pair_path,
                                3ULL * step
                            );
                    }
                    const std::size_t second_path = first_pair_path + 1U;
                    if (second_path < range.paths_per_parameter
                        && contains_sample(range, first_sample + 1U)) {
                        increment.y = row.sqrt_time_step
                            * volterra::hybrid_fft::normal_at(
                                row.dynamics_key,
                                second_path,
                                3ULL * step
                            );
                    }
                    return increment;
                }
            };
            struct Store {
                float2* outputs;
                unsigned int local_fft;

                __device__ __forceinline__ void operator()(
                    unsigned int index,
                    float2 value
                ) const {
                    outputs[
                        static_cast<std::size_t>(local_fft) * Length + index
                    ] = value;
                }
            };

            volterra::execute_padded_linear_convolution<
                Length,
                Forward,
                Inverse
            >(
                kernel_spectrum,
                Loader{
                    row,
                    range,
                    parameter_sample_begin,
                    first_pair_path,
                    maximum_step_count,
                },
                Store{packed_outputs, local_fft},
                execution_storage
            );
            __syncthreads();

            if (threadIdx.x < 2U && pair < pair_count) {
                const std::size_t path_index =
                    first_pair_path + threadIdx.x;
                const std::size_t sample_index =
                    parameter_sample_begin + path_index;
                if (path_index < range.paths_per_parameter
                    && contains_sample(range, sample_index)) {
                    const typename Schedule::Calendar calendar =
                        calendar_source.load(sample_index, seeds.schedule);
                    const std::uint64_t path_steps =
                        kSampleSimulationStepsPerDay
                        * maturity_days(calendar);
                    if (path_steps > 0U
                        && path_steps <= maximum_step_count) {
                        const std::uint32_t path_step_count =
                            static_cast<std::uint32_t>(path_steps);
                        const volterra::HybridTimeConfiguration
                            time_configuration{
                                kSampleDayFraction,
                                kSampleFixedStepDt,
                            };
                        const typename Schedule::PreparedSchedule schedule =
                            Schedule::prepare(
                                calendar,
                                time_configuration,
                                path_step_count
                            );
                        typename Path::State state =
                            Path::initial_state(row.model);
                        typename Observation::Handler handler =
                            Observation::make_handler(
                                output,
                                sample_index,
                                total_samples
                            );
                        typename Schedule::Cursor cursor =
                            Schedule::make_cursor(schedule);
                        bool keep_running = Schedule::on_initial_state(
                            schedule,
                            cursor,
                            state,
                            handler
                        );
                        for (std::uint32_t step = 0U;
                             keep_running && step < path_step_count;
                             ++step) {
                            const float rough_normal =
                                volterra::hybrid_fft::normal_at(
                                    row.dynamics_key,
                                    path_index,
                                    3ULL * step
                                );
                            const float singular_normal =
                                volterra::hybrid_fft::normal_at(
                                    row.dynamics_key,
                                    path_index,
                                    3ULL * step + 1ULL
                                );
                            const float spot_normal =
                                volterra::hybrid_fft::normal_at(
                                    row.dynamics_key,
                                    path_index,
                                    3ULL * step + 2ULL
                                );
                            float far_convolution = 0.0f;
                            if (step > 0U) {
                                const float2 packed = packed_outputs[
                                    static_cast<std::size_t>(local_fft)
                                        * Length
                                        + step - 1U
                                ];
                                far_convolution = threadIdx.x == 0U
                                    ? packed.x
                                    : packed.y;
                            }
                            const float volterra_value =
                                Kernel::reconstruct_volterra_value(
                                    row.kernel,
                                    far_convolution,
                                    rough_normal,
                                    singular_normal
                                );
                            Path::advance(
                                row.model,
                                volterra_value,
                                volterra_variances[step],
                                rough_normal,
                                spot_normal,
                                state
                            );
                            keep_running = Schedule::on_step(
                                schedule,
                                cursor,
                                step,
                                state,
                                handler
                            );
                        }
                    }
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

template<
    typename Policy,
    ParameterSourceFor<typename Policy::Parameters> ParameterSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
requires VolterraFftSamplingPolicy<
    Policy,
    volterra::HybridTimeConfiguration
>
void validate_launch(
    const ParameterSource& parameter_source,
    const CalendarSource& calendar_source,
    const SampleRange& range,
    LaunchConfiguration launch_configuration,
    SamplingSeeds seeds,
    const typename Policy::Output& output
) {
    const std::size_t total_samples = total_sample_count(range);
    if (range.sample_offset >= total_samples
        || range.launch_sample_count == 0U
        || range.launch_sample_count
            > total_samples - range.sample_offset) {
        throw std::invalid_argument(
            "The Volterra sample batch exceeds the logical sample array."
        );
    }
    if (launch_configuration.maximum_step_count == 0U
        || launch_configuration.maximum_step_count
            % kSampleSimulationStepsPerDay != 0U
        || launch_configuration.maximum_step_count
            > kMaximumSampleStepCount) {
        throw std::invalid_argument(
            "Volterra FFT sample maximum_step_count must be an even value "
            "in [2, 1008]."
        );
    }
    parameter_source.validate(range.parameter_count);
    calendar_source.validate(total_samples);
    if constexpr (requires { calendar_source.maximum_maturity_days(); }) {
        const std::uint64_t required_steps =
            kSampleSimulationStepsPerDay
            * static_cast<std::uint64_t>(
                calendar_source.maximum_maturity_days()
            );
        if (required_steps > launch_configuration.maximum_step_count) {
            throw std::invalid_argument(
                "The Volterra FFT length does not cover the generated "
                "calendar support."
            );
        }
    }
    Policy::Observation::validate(output);
    validate_row_seed_range(range.parameter_count, seeds.dynamics);
    validate_grid_x_size(launch_configuration.block_count);
    validate_block_count(
        launch_parameter_count(range),
        launch_configuration.block_count
    );
}

template<
    typename Policy,
    ParameterSourceFor<typename Policy::Parameters> ParameterSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource,
    unsigned int Length,
    unsigned int ElementsPerThread,
    unsigned int FftsPerBlock
>
requires VolterraFftSamplingPolicy<
    Policy,
    volterra::HybridTimeConfiguration
>
void launch_fft_length(
    const ParameterSource& parameter_source,
    const CalendarSource& calendar_source,
    const SampleRange& range,
    LaunchConfiguration launch_configuration,
    SamplingSeeds seeds,
    const typename Policy::Output& output,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name
) {
    using ExecutionTypes = volterra::hybrid_fft::FftTypes<
        Length,
        ElementsPerThread,
        FftsPerBlock
    >;
    using Forward = typename ExecutionTypes::Forward;
    using Inverse = typename ExecutionTypes::Inverse;
    static_assert(!Forward::requires_workspace);
    static_assert(!Inverse::requires_workspace);

    constexpr std::size_t spectrum_bytes =
        static_cast<std::size_t>(Length) * sizeof(float2);
    constexpr std::size_t packed_output_bytes =
        static_cast<std::size_t>(Length)
        * FftsPerBlock
        * sizeof(float2);
    constexpr std::size_t transform_bytes =
        static_cast<std::size_t>(Forward::shared_memory_size)
            > static_cast<std::size_t>(Inverse::shared_memory_size)
        ? static_cast<std::size_t>(Forward::shared_memory_size)
        : static_cast<std::size_t>(Inverse::shared_memory_size);
    constexpr std::size_t execution_bytes =
        transform_bytes > packed_output_bytes
        ? transform_bytes
        : packed_output_bytes;
    const std::size_t shared_bytes = spectrum_bytes
        + execution_bytes
        + static_cast<std::size_t>(launch_configuration.maximum_step_count)
            * sizeof(float);
    const auto kernel = parameter_block_sample_kernel<
        Policy,
        ParameterSource,
        CalendarSource,
        Length,
        Forward,
        Inverse
    >;
    check_cuda(
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)
        ),
        "Volterra FFT sample shared-memory opt-in"
    );
    int active_blocks = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,
            kernel,
            static_cast<int>(Forward::block_dim.x * Forward::block_dim.y),
            shared_bytes
        ),
        "Volterra FFT sample occupancy check"
    );
    if (active_blocks == 0) {
        throw std::invalid_argument(
            "The Volterra FFT sample kernel cannot reside on an SM."
        );
    }

    const dim3 grid(
        static_cast<unsigned int>(launch_configuration.block_count)
    );
    report_cuda_kernel_launch_if_enabled(
        diagnostic_name,
        diagnostic_variant,
        kernel,
        grid,
        Forward::block_dim,
        shared_bytes
    );
    kernel<<<grid, Forward::block_dim, shared_bytes>>>(
        parameter_source,
        calendar_source,
        range,
        total_sample_count(range),
        launch_configuration.maximum_step_count,
        seeds,
        output
    );
    check_cuda(cudaGetLastError(), operation_name);
}

template<
    typename Policy,
    ParameterSourceFor<typename Policy::Parameters> ParameterSource,
    CalendarSourceFor<typename Policy::Schedule::Calendar> CalendarSource
>
requires VolterraFftSamplingPolicy<
    Policy,
    volterra::HybridTimeConfiguration
>
void launch_samples_cuda(
    const ParameterSource& parameter_source,
    const CalendarSource& calendar_source,
    const SampleRange& range,
    LaunchConfiguration launch_configuration,
    SamplingSeeds seeds,
    const typename Policy::Output& output,
    const char* diagnostic_name,
    const char* diagnostic_variant,
    const char* operation_name
) {
    validate_launch<Policy>(
        parameter_source,
        calendar_source,
        range,
        launch_configuration,
        seeds,
        output
    );
    const std::uint32_t steps = launch_configuration.maximum_step_count;
    if (steps <= 8U) {
        launch_fft_length<Policy, ParameterSource, CalendarSource, 16U, 8U, 16U>(
            parameter_source, calendar_source, range, launch_configuration,
            seeds, output, diagnostic_name, diagnostic_variant, operation_name
        );
    } else if (steps <= 32U) {
        launch_fft_length<Policy, ParameterSource, CalendarSource, 64U, 8U, 8U>(
            parameter_source, calendar_source, range, launch_configuration,
            seeds, output, diagnostic_name, diagnostic_variant, operation_name
        );
    } else if (steps <= 64U) {
        launch_fft_length<Policy, ParameterSource, CalendarSource, 128U, 8U, 8U>(
            parameter_source, calendar_source, range, launch_configuration,
            seeds, output, diagnostic_name, diagnostic_variant, operation_name
        );
    } else if (steps <= 128U) {
        launch_fft_length<Policy, ParameterSource, CalendarSource, 256U, 16U, 8U>(
            parameter_source, calendar_source, range, launch_configuration,
            seeds, output, diagnostic_name, diagnostic_variant, operation_name
        );
    } else if (steps <= 256U) {
        launch_fft_length<Policy, ParameterSource, CalendarSource, 512U, 8U, 2U>(
            parameter_source, calendar_source, range, launch_configuration,
            seeds, output, diagnostic_name, diagnostic_variant, operation_name
        );
    } else if (steps <= 512U) {
        launch_fft_length<Policy, ParameterSource, CalendarSource, 1024U, 16U, 1U>(
            parameter_source, calendar_source, range, launch_configuration,
            seeds, output, diagnostic_name, diagnostic_variant, operation_name
        );
    } else {
        launch_fft_length<Policy, ParameterSource, CalendarSource, 2048U, 16U, 1U>(
            parameter_source, calendar_source, range, launch_configuration,
            seeds, output, diagnostic_name, diagnostic_variant, operation_name
        );
    }
}

template<typename Policy>
requires VolterraFftSamplingPolicy<
    Policy,
    volterra::HybridTimeConfiguration
>
void launch_device_terminal_samples_cuda(
    const typename Policy::Parameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t maturity_days,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::size_t block_count,
    std::uint64_t dynamics_seed,
    const typename Policy::Output& output,
    const char* diagnostic_name,
    const char* operation_name
) {
    static_assert(std::same_as<
        typename Policy::Schedule::Calendar,
        simulation::MaturityCalendar
    >);
    launch_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Parameters>{device_parameters},
        ConstantCalendarSource<simulation::MaturityCalendar>{{maturity_days}},
        {
            parameter_count,
            paths_per_parameter,
            sample_offset,
            launch_sample_count,
        },
        {canonical_sample_step_count(maturity_days), block_count},
        {0U, 0U, dynamics_seed},
        output,
        diagnostic_name,
        "terminal.parameter_block_fft",
        operation_name
    );
}

template<typename Policy>
requires VolterraFftSamplingPolicy<
    Policy,
    volterra::HybridTimeConfiguration
>
void launch_device_random_terminal_samples_cuda(
    const typename Policy::Parameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    MaturityDayBounds maturity_bounds,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::size_t block_count,
    std::uint64_t schedule_seed,
    std::uint64_t dynamics_seed,
    std::uint32_t* device_maturity_days,
    const typename Policy::Output& output,
    const char* diagnostic_name,
    const char* operation_name
) {
    static_assert(std::same_as<
        typename Policy::Schedule::Calendar,
        simulation::MaturityCalendar
    >);
    launch_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Parameters>{device_parameters},
        UniformMaturityCalendarSource{
            maturity_bounds,
            device_maturity_days,
        },
        {
            parameter_count,
            paths_per_parameter,
            sample_offset,
            launch_sample_count,
        },
        {
            canonical_sample_step_count(maturity_bounds.maximum),
            block_count,
        },
        {0U, schedule_seed, dynamics_seed},
        output,
        diagnostic_name,
        "random_terminal.parameter_block_fft",
        operation_name
    );
}

template<typename Policy>
requires VolterraFftSamplingPolicy<
    Policy,
    volterra::HybridTimeConfiguration
>
void launch_device_calendar_samples_cuda(
    const typename Policy::Parameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    std::size_t block_count,
    std::uint64_t dynamics_seed,
    const typename Policy::Output& output,
    const char* diagnostic_name,
    const char* operation_name
) {
    static_assert(std::same_as<
        typename Policy::Schedule::Calendar,
        simulation::StubbedRegularCalendar
    >);
    const simulation::StubbedRegularCalendar calendar{
        first_observation_day,
        observation_interval_days,
        observation_count,
    };
    launch_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Parameters>{device_parameters},
        ConstantCalendarSource<simulation::StubbedRegularCalendar>{calendar},
        {
            parameter_count,
            paths_per_parameter,
            sample_offset,
            launch_sample_count,
        },
        {
            canonical_sample_calendar_step_count(
                first_observation_day,
                observation_interval_days,
                observation_count
            ),
            block_count,
        },
        {0U, 0U, dynamics_seed},
        output,
        diagnostic_name,
        "calendar.parameter_block_fft",
        operation_name
    );
}

}  // namespace ai_factory::workbench::sample::volterra_fft
