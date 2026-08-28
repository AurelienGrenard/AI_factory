// Compile-time contracts and runtime descriptors for model-only sampling.
#pragma once

#include "common/philox.cuh"
#include "common/simulation/concepts.cuh"
#include "common/volterra/concepts.cuh"

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::sample {

struct SamplingSeeds {
    std::uint64_t parameters;
    std::uint64_t schedule;
    std::uint64_t dynamics;
};

struct SampleRange {
    std::size_t parameter_count;
    std::size_t paths_per_parameter;
    std::size_t sample_offset;
    std::size_t launch_sample_count;
};

struct SampleLaunchConfiguration {
    unsigned int threads_per_block;
    std::size_t block_count;
};

enum class SampleExecutionStrategy : std::uint8_t {
    automatic,
    thread_grid_stride,
    parameter_block,
};

template<typename Source, typename Parameters>
concept ParameterSourceFor =
    std::is_trivially_copyable_v<Source>
    && requires(
        const Source& source,
        std::size_t parameter_index,
        std::size_t parameter_count,
        std::uint64_t seed
    ) {
        { source.validate(parameter_count) } -> std::same_as<void>;
        {
            source.load(parameter_index, seed)
        } -> std::same_as<Parameters>;
    };

template<typename Source, typename Calendar>
concept CalendarSourceFor =
    std::is_trivially_copyable_v<Source>
    && requires(
        const Source& source,
        std::size_t sample_index,
        std::size_t total_sample_count,
        std::uint64_t seed
    ) {
        { source.validate(total_sample_count) } -> std::same_as<void>;
        {
            source.load(sample_index, seed)
        } -> std::same_as<Calendar>;
    };

template<typename Sampler>
concept ParameterSamplingPolicy =
    std::is_trivially_copyable_v<typename Sampler::Configuration>
    && std::is_trivially_copyable_v<typename Sampler::Parameters>
    && requires(
        const typename Sampler::Configuration& configuration,
        philox::PhiloxKey key,
        std::uint64_t parameter_index
    ) {
        { Sampler::validate(configuration) } -> std::same_as<void>;
        {
            Sampler::sample(configuration, key, parameter_index)
        } -> std::same_as<typename Sampler::Parameters>;
    };

template<typename Observation, typename Dynamics>
concept SampleObservationPolicyFor =
    simulation::StatePolicy<Dynamics>
    && std::is_trivially_copyable_v<typename Observation::Output>
    && std::is_trivially_copyable_v<typename Observation::Handler>
    && simulation::ObservationHandlerFor<
        typename Observation::Handler,
        Dynamics
    >
    && requires(
        const typename Observation::Output& output,
        const typename Dynamics::State& state,
        std::size_t sample_index,
        std::size_t total_sample_count
    ) {
        { Observation::validate(output) } -> std::same_as<void>;
        {
            Observation::write_terminal(output, sample_index, state)
        } -> std::same_as<void>;
        {
            Observation::make_handler(
                output,
                sample_index,
                total_sample_count
            )
        } -> std::same_as<typename Observation::Handler>;
    };

template<typename SchedulePolicy, typename ObservationPolicy>
requires SampleObservationPolicyFor<
    ObservationPolicy,
    typename SchedulePolicy::Dynamics
>
struct ModelSamplingPolicy {
    using Schedule = SchedulePolicy;
    using Dynamics = typename Schedule::Dynamics;
    using Parameters = typename Dynamics::Parameters;
    using Observation = ObservationPolicy;
    using Output = typename Observation::Output;
};

template<typename Policy>
concept SamplingPolicy =
    simulation::SchedulePolicy<typename Policy::Schedule>
    && simulation::DevicePreparedSchedulePolicy<typename Policy::Schedule>
    && simulation::ReusablePreparedInputSchedulePolicy<
        typename Policy::Schedule
    >
    && SampleObservationPolicyFor<
        typename Policy::Observation,
        typename Policy::Schedule::Dynamics
    >
    && (simulation::TerminalSchedulePolicy<typename Policy::Schedule>
        || simulation::ObservedSchedulePolicy<typename Policy::Schedule>);

// Some Markov schemes prepare expensive numerical coefficients on the host.
// They still use the ordinary schedule and path executor, but their sampling
// source supplies Schedule::PreparedInput directly instead of raw parameters.
template<typename Policy>
concept ExternallyPreparedSamplingPolicy =
    simulation::SchedulePolicy<typename Policy::Schedule>
    && simulation::ExternallyPreparedSchedulePolicy<
        typename Policy::Schedule
    >
    && std::is_trivially_copyable_v<
        typename Policy::Schedule::PreparedInput
    >
    && SampleObservationPolicyFor<
        typename Policy::Observation,
        typename Policy::Schedule::Dynamics
    >
    && requires(
        const typename Policy::Schedule::PreparedInput& prepared_input,
        const typename Policy::Schedule::Calendar& calendar,
        const typename Policy::Schedule::TimeConfiguration& time_configuration
    ) {
        {
            Policy::Schedule::prepare_from_input(
                prepared_input,
                calendar,
                time_configuration
            )
        } -> std::same_as<typename Policy::Schedule::PreparedSchedule>;
    }
    && (simulation::TerminalSchedulePolicy<typename Policy::Schedule>
        || simulation::ObservedSchedulePolicy<typename Policy::Schedule>);

template<typename Policy>
concept ExecutableSamplingPolicy =
    SamplingPolicy<Policy> || ExternallyPreparedSamplingPolicy<Policy>;

template<typename Path>
concept VolterraPathPolicy =
    volterra::HybridPathPolicy<Path>;

template<typename Kernel>
concept VolterraKernelPolicy =
    volterra::HybridKernelPolicy<Kernel>;

template<typename Schedule, typename Path, typename Handler,
         typename TimeConfiguration>
concept VolterraSampleSchedulePolicy =
    std::is_trivially_copyable_v<typename Schedule::Calendar>
    && std::is_trivially_copyable_v<typename Schedule::PreparedSchedule>
    && std::is_trivially_copyable_v<typename Schedule::Cursor>
    && requires(
    const typename Schedule::Calendar& calendar,
    const typename Schedule::PreparedSchedule& schedule,
    typename Schedule::Cursor& cursor,
    const typename Path::State& state,
    Handler& handler,
    const TimeConfiguration& time_configuration,
    std::uint32_t step_count,
    std::uint32_t step
) {
    {
        Schedule::prepare(calendar, time_configuration, step_count)
    } -> std::same_as<typename Schedule::PreparedSchedule>;
    {
        Schedule::make_cursor(schedule)
    } -> std::same_as<typename Schedule::Cursor>;
    {
        Schedule::on_initial_state(
            schedule, cursor, state, handler
        )
    } -> std::same_as<bool>;
    {
        Schedule::on_step(schedule, cursor, step, state, handler)
    } -> std::same_as<bool>;
};

template<typename KernelPolicy, typename PathPolicy, typename SchedulePolicy,
         typename ObservationPolicy>
requires (
    VolterraKernelPolicy<KernelPolicy>
    && VolterraPathPolicy<PathPolicy>
    && volterra::HybridPathPolicyFor<PathPolicy, KernelPolicy>
    && SampleObservationPolicyFor<ObservationPolicy, PathPolicy>
)
struct VolterraFftModelSamplingPolicy {
    using Kernel = KernelPolicy;
    using Path = PathPolicy;
    using Parameters = typename Path::Parameters;
    using Schedule = SchedulePolicy;
    using Observation = ObservationPolicy;
    using Output = typename Observation::Output;
};

template<typename Policy, typename TimeConfiguration>
concept VolterraFftSamplingPolicy =
    VolterraKernelPolicy<typename Policy::Kernel>
    && VolterraPathPolicy<typename Policy::Path>
    && SampleObservationPolicyFor<
        typename Policy::Observation,
        typename Policy::Path
    >
    && VolterraSampleSchedulePolicy<
        typename Policy::Schedule,
        typename Policy::Path,
        typename Policy::Observation::Handler,
        TimeConfiguration
    >;

}  // namespace ai_factory::workbench::sample
