// Schobel-Zhu composition over the common model-sampling engine.
#include "model/equity/markovian/schobel_zhu/sample.cuh"
#include "common/sample.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"
namespace ai_factory::workbench::model::equity::schobel_zhu {
namespace {
using Observation = sample::SpotAndStateSampleObservation<
    DynamicsPolicy, &State::volatility>;
using TerminalPolicy = sample::ModelSamplingPolicy<
    simulation::FixedStepTerminalSchedule<DynamicsPolicy>, Observation>;
using CalendarPolicy = sample::ModelSamplingPolicy<
    simulation::FixedStepStubbedRegularSchedule<DynamicsPolicy>, Observation>;
static_assert(sample::SamplingPolicy<TerminalPolicy>);
static_assert(sample::SamplingPolicy<CalendarPolicy>);
}
void launch_schobel_zhu_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t maturity_days,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed, float* device_spots,
    float* device_volatilities
) {
    sample::launch_device_terminal_samples_cuda<TerminalPolicy>(
        device_parameters, parameter_count, paths_per_parameter, maturity_days,
        sample_offset, launch_sample_count, threads_per_block, block_count,
        dynamics_seed, {device_spots, device_volatilities},
        "schobel_zhu.samples",
        "Schobel-Zhu terminal sample kernel");
}
void launch_schobel_zhu_calendar_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days, std::uint32_t observation_count,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed, float* device_spots,
    float* device_volatilities
) {
    sample::launch_device_calendar_samples_cuda<CalendarPolicy>(
        device_parameters, parameter_count, paths_per_parameter,
        first_observation_day, observation_interval_days, observation_count,
        sample_offset, launch_sample_count, threads_per_block, block_count,
        dynamics_seed, {device_spots, device_volatilities},
        "schobel_zhu.samples",
        "Schobel-Zhu calendar sample kernel");
}
}  // namespace ai_factory::workbench::model::equity::schobel_zhu
