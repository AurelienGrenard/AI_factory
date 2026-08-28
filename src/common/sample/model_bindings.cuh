// Thin model-binding helpers for caller-provided parameters and day calendars.
#pragma once

#include "common/sample/sample_kernels.cuh"
#include "common/sample/sources.cuh"
#include "common/simulation/schedule.cuh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::sample {

inline constexpr float kSampleDayFraction = 1.0f / 252.0f;
inline constexpr float kSampleFixedStepDt = 1.0f / 504.0f;
inline constexpr std::uint32_t kSampleSimulationStepsPerDay = 2U;

inline constexpr simulation::ExactTransitionTimeConfiguration
    kExactSampleTimeConfiguration{kSampleDayFraction};
inline constexpr simulation::FixedStepTimeConfiguration
    kFixedStepSampleTimeConfiguration{
        kSampleFixedStepDt,
        kSampleSimulationStepsPerDay,
    };

inline std::uint32_t canonical_sample_step_count(
    std::uint32_t maturity_days
) {
    if (maturity_days == 0U
        || maturity_days
            > std::numeric_limits<std::uint32_t>::max()
                / kSampleSimulationStepsPerDay) {
        throw std::invalid_argument(
            "A sample maturity must map to a positive uint32_t step count."
        );
    }
    return kSampleSimulationStepsPerDay * maturity_days;
}

inline std::uint32_t canonical_sample_calendar_step_count(
    std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days,
    std::uint32_t observation_count
) {
    simulation::validate_calendar(simulation::StubbedRegularCalendar{
        first_observation_day,
        observation_interval_days,
        observation_count,
    });
    const std::uint64_t maturity_days = first_observation_day
        + static_cast<std::uint64_t>(observation_count - 1U)
            * observation_interval_days;
    if (maturity_days
        > std::numeric_limits<std::uint32_t>::max()
            / kSampleSimulationStepsPerDay) {
        throw std::overflow_error(
            "The sample calendar step count exceeds uint32_t."
        );
    }
    return static_cast<std::uint32_t>(
        kSampleSimulationStepsPerDay * maturity_days
    );
}

template<typename TimeConfiguration>
constexpr TimeConfiguration canonical_sample_time_configuration() {
    if constexpr (std::same_as<
        TimeConfiguration,
        simulation::ExactTransitionTimeConfiguration
    >) {
        return kExactSampleTimeConfiguration;
    } else {
        static_assert(std::same_as<
            TimeConfiguration,
            simulation::FixedStepTimeConfiguration
        >);
        return kFixedStepSampleTimeConfiguration;
    }
}

template<SamplingPolicy Policy>
inline void launch_device_terminal_samples_cuda(
    const typename Policy::Parameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t maturity_days,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
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
    const SampleRange range{
        parameter_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
    };
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        SampleExecutionStrategy::automatic,
        paths_per_parameter
    );
    launch_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Parameters>{device_parameters},
        ConstantCalendarSource<simulation::MaturityCalendar>{
            {maturity_days}
        },
        range,
        canonical_sample_time_configuration<
            typename Policy::Schedule::TimeConfiguration
        >(),
        {threads_per_block, block_count},
        {0U, 0U, dynamics_seed},
        output,
        diagnostic_name,
        strategy == SampleExecutionStrategy::parameter_block
            ? "terminal.parameter_block"
            : "terminal.thread_grid_stride",
        operation_name,
        strategy
    );
}

template<SamplingPolicy Policy>
inline void launch_device_random_terminal_samples_cuda(
    const typename Policy::Parameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    MaturityDayBounds maturity_bounds,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
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
    const SampleRange range{
        parameter_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
    };
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        SampleExecutionStrategy::automatic,
        paths_per_parameter
    );
    launch_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Parameters>{device_parameters},
        UniformMaturityCalendarSource{
            maturity_bounds,
            device_maturity_days,
        },
        range,
        canonical_sample_time_configuration<
            typename Policy::Schedule::TimeConfiguration
        >(),
        {threads_per_block, block_count},
        {0U, schedule_seed, dynamics_seed},
        output,
        diagnostic_name,
        strategy == SampleExecutionStrategy::parameter_block
            ? "random_terminal.parameter_block"
            : "random_terminal.thread_grid_stride",
        operation_name,
        strategy
    );
}

template<SamplingPolicy Policy>
inline void launch_device_calendar_samples_cuda(
    const typename Policy::Parameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
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
    const SampleRange range{
        parameter_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
    };
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        SampleExecutionStrategy::automatic,
        paths_per_parameter
    );
    launch_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Parameters>{device_parameters},
        ConstantCalendarSource<simulation::StubbedRegularCalendar>{
            {
                first_observation_day,
                observation_interval_days,
                observation_count,
            }
        },
        range,
        canonical_sample_time_configuration<
            typename Policy::Schedule::TimeConfiguration
        >(),
        {threads_per_block, block_count},
        {0U, 0U, dynamics_seed},
        output,
        diagnostic_name,
        strategy == SampleExecutionStrategy::parameter_block
            ? "calendar.parameter_block"
            : "calendar.thread_grid_stride",
        operation_name,
        strategy
    );
}

template<ExternallyPreparedSamplingPolicy Policy>
inline void launch_device_prepared_terminal_samples_cuda(
    const typename Policy::Schedule::PreparedInput* device_prepared_inputs,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t maturity_days,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
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
    const SampleRange range{
        parameter_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
    };
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        SampleExecutionStrategy::automatic,
        paths_per_parameter
    );
    launch_prepared_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Schedule::PreparedInput>{
            device_prepared_inputs
        },
        ConstantCalendarSource<simulation::MaturityCalendar>{
            {maturity_days}
        },
        range,
        canonical_sample_time_configuration<
            typename Policy::Schedule::TimeConfiguration
        >(),
        {threads_per_block, block_count},
        {0U, 0U, dynamics_seed},
        output,
        diagnostic_name,
        strategy == SampleExecutionStrategy::parameter_block
            ? "terminal.prepared_parameter_block"
            : "terminal.prepared_thread_grid_stride",
        operation_name,
        strategy
    );
}

template<ExternallyPreparedSamplingPolicy Policy>
inline void launch_device_prepared_random_terminal_samples_cuda(
    const typename Policy::Schedule::PreparedInput* device_prepared_inputs,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    MaturityDayBounds maturity_bounds,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
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
    const SampleRange range{
        parameter_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
    };
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        SampleExecutionStrategy::automatic,
        paths_per_parameter
    );
    launch_prepared_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Schedule::PreparedInput>{
            device_prepared_inputs
        },
        UniformMaturityCalendarSource{
            maturity_bounds,
            device_maturity_days,
        },
        range,
        canonical_sample_time_configuration<
            typename Policy::Schedule::TimeConfiguration
        >(),
        {threads_per_block, block_count},
        {0U, schedule_seed, dynamics_seed},
        output,
        diagnostic_name,
        strategy == SampleExecutionStrategy::parameter_block
            ? "random_terminal.prepared_parameter_block"
            : "random_terminal.prepared_thread_grid_stride",
        operation_name,
        strategy
    );
}

template<ExternallyPreparedSamplingPolicy Policy>
inline void launch_device_prepared_calendar_samples_cuda(
    const typename Policy::Schedule::PreparedInput* device_prepared_inputs,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days,
    std::uint32_t observation_count,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
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
    const SampleRange range{
        parameter_count,
        paths_per_parameter,
        sample_offset,
        launch_sample_count,
    };
    const SampleExecutionStrategy strategy = resolve_execution_strategy(
        SampleExecutionStrategy::automatic,
        paths_per_parameter
    );
    launch_prepared_samples_cuda<Policy>(
        DeviceParameterSource<typename Policy::Schedule::PreparedInput>{
            device_prepared_inputs
        },
        ConstantCalendarSource<simulation::StubbedRegularCalendar>{
            {
                first_observation_day,
                observation_interval_days,
                observation_count,
            }
        },
        range,
        canonical_sample_time_configuration<
            typename Policy::Schedule::TimeConfiguration
        >(),
        {threads_per_block, block_count},
        {0U, 0U, dynamics_seed},
        output,
        diagnostic_name,
        strategy == SampleExecutionStrategy::parameter_block
            ? "calendar.prepared_parameter_block"
            : "calendar.prepared_thread_grid_stride",
        operation_name,
        strategy
    );
}

}  // namespace ai_factory::workbench::sample
