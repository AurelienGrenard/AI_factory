// Generated SABR composition over the common sample engine.
#include "model/equity/markovian/sabr/sample.cuh"
#include "model/equity/markovian/sabr/sample_parameter_sampler.cuh"

#include <sample.cuh>
#include "schedule/schedule.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::sabr {
namespace {
using State = typename DynamicsPolicy::State;
using Observation = sample::SpotAndStateSampleObservation<DynamicsPolicy, &State::alpha>;
using TerminalPolicy = sample::ModelSamplingPolicy<
    simulation::FixedStepTerminalSchedule<DynamicsPolicy>, Observation>;
using CalendarPolicy = sample::ModelSamplingPolicy<
    simulation::FixedStepStubbedRegularSchedule<DynamicsPolicy>, Observation>;
static_assert(sample::SamplingPolicy<TerminalPolicy>);
static_assert(sample::SamplingPolicy<CalendarPolicy>);
}  // namespace

void launch_sabr_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t maturity_days,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed,
    float* device_spots,
    float* device_alphas
) {
    sample::launch_device_terminal_samples_cuda<TerminalPolicy>(
        device_parameters, parameter_count, paths_per_parameter, maturity_days,
        sample_offset, launch_sample_count, threads_per_block, block_count,
        dynamics_seed, {device_spots, device_alphas}, "sabr.samples",
        "SABR terminal sample kernel");
}

void launch_sabr_random_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t schedule_seed,
    std::uint64_t dynamics_seed, std::uint32_t* device_maturity_days
    ,
    float* device_spots,
    float* device_alphas
) {
    sample::launch_device_random_terminal_samples_cuda<TerminalPolicy>(
        device_parameters, parameter_count, paths_per_parameter,
        {minimum_maturity_days, maximum_maturity_days}, sample_offset,
        launch_sample_count, threads_per_block, block_count, schedule_seed,
        dynamics_seed, device_maturity_days, {device_spots, device_alphas},
        "sabr.samples", "SABR random terminal sample kernel");
}

void launch_sabr_generated_random_terminal_samples_cuda(
    ModelParameters* device_generated_parameters,
    SampleParameterSampler::Configuration bounds,
    std::size_t parameter_count, std::size_t paths_per_parameter,
    std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t parameter_seed,
    std::uint64_t schedule_seed, std::uint64_t dynamics_seed,
    std::uint32_t* device_maturity_days,
    float* device_spots,
    float* device_alphas
) {
    sample::launch_samples_cuda<TerminalPolicy>(
        sample::GeneratedParameterSource<SampleParameterSampler>{
            bounds, device_generated_parameters},
        sample::UniformMaturityCalendarSource{
            {minimum_maturity_days, maximum_maturity_days},
            device_maturity_days},
        {parameter_count, paths_per_parameter, sample_offset,
         launch_sample_count},
        sample::canonical_sample_time_configuration<
            typename TerminalPolicy::Schedule::TimeConfiguration>(),
        {threads_per_block, block_count},
        {parameter_seed, schedule_seed, dynamics_seed},
        {device_spots, device_alphas}, "sabr.samples",
        "random_terminal.generated_parameters",
        "SABR generated random terminal sample kernel");
}

void launch_sabr_calendar_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days, std::uint32_t observation_count,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed,
    float* device_spots,
    float* device_alphas
) {
    sample::launch_device_calendar_samples_cuda<CalendarPolicy>(
        device_parameters, parameter_count, paths_per_parameter,
        first_observation_day, observation_interval_days, observation_count,
        sample_offset, launch_sample_count, threads_per_block, block_count,
        dynamics_seed, {device_spots, device_alphas}, "sabr.samples",
        "SABR calendar sample kernel");
}

}  // namespace ai_factory::workbench::model::equity::sabr
