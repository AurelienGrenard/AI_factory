// Generated Rough Stein-Stein composition over the shared FFT sampler.
#include "model/equity/rough/rough_stein_stein/sample.cuh"

#include "common/sample.cuh"
#include "common/volterra/fractional_resolvent_hybrid_driver.cuh"
#include "common/volterra/hybrid_schedule.cuh"
#include "model/equity/rough/rough_stein_stein/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::rough_stein_stein {
namespace {
using Observation = sample::SpotSampleObservation<PathPolicy>;
using TerminalPolicy = sample::VolterraFftModelSamplingPolicy<
    volterra::FractionalResolventHybridDriverPolicy, PathPolicy, volterra::TerminalHybridSchedule, Observation>;
using CalendarPolicy = sample::VolterraFftModelSamplingPolicy<
    volterra::FractionalResolventHybridDriverPolicy, PathPolicy, volterra::StubbedRegularHybridSchedule, Observation>;
static_assert(sample::VolterraFftSamplingPolicy<
    TerminalPolicy, volterra::HybridTimeConfiguration>);
static_assert(sample::VolterraFftSamplingPolicy<
    CalendarPolicy, volterra::HybridTimeConfiguration>);
}  // namespace

void launch_rough_stein_stein_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t maturity_days,
    std::size_t sample_offset, std::size_t launch_sample_count,
    std::size_t block_count, std::uint64_t dynamics_seed
    ,
    float* device_spots
) {
    sample::volterra_fft::launch_device_terminal_samples_cuda<TerminalPolicy>(
        device_parameters, parameter_count, paths_per_parameter, maturity_days,
        sample_offset, launch_sample_count, block_count, dynamics_seed,
        {device_spots}, "rough_stein_stein.samples",
        "Rough Stein-Stein terminal FFT sample kernel");
}

void launch_rough_stein_stein_random_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, std::size_t block_count,
    std::uint64_t schedule_seed, std::uint64_t dynamics_seed,
    std::uint32_t* device_maturity_days,
    float* device_spots
) {
    sample::volterra_fft::launch_device_random_terminal_samples_cuda<
        TerminalPolicy>(
        device_parameters, parameter_count, paths_per_parameter,
        {minimum_maturity_days, maximum_maturity_days}, sample_offset,
        launch_sample_count, block_count, schedule_seed, dynamics_seed,
        device_maturity_days, {device_spots}, "rough_stein_stein.samples",
        "Rough Stein-Stein random terminal FFT sample kernel");
}

void launch_rough_stein_stein_calendar_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days, std::uint32_t observation_count,
    std::size_t sample_offset, std::size_t launch_sample_count,
    std::size_t block_count, std::uint64_t dynamics_seed
    ,
    float* device_spots
) {
    sample::volterra_fft::launch_device_calendar_samples_cuda<CalendarPolicy>(
        device_parameters, parameter_count, paths_per_parameter,
        first_observation_day, observation_interval_days, observation_count,
        sample_offset, launch_sample_count, block_count, dynamics_seed,
        {device_spots}, "rough_stein_stein.samples",
        "Rough Stein-Stein calendar FFT sample kernel");
}

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
