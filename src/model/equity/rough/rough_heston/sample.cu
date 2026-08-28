// Generated Rough-Heston composition over the prepared Markovian N-factor sample engine.
#include "model/equity/rough/rough_heston/sample.cuh"

#include "common/sample.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/rough/rough_heston/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::rough_heston {
namespace {
template<std::size_t FactorCount>
using Observation = sample::SpotSampleObservation<DynamicsPolicy<FactorCount>>;
template<std::size_t FactorCount>
using TerminalPolicy = sample::ModelSamplingPolicy<
    simulation::FixedStepTerminalSchedule<DynamicsPolicy<FactorCount>>,
    Observation<FactorCount>>;
template<std::size_t FactorCount>
using CalendarPolicy = sample::ModelSamplingPolicy<
    simulation::FixedStepStubbedRegularSchedule<DynamicsPolicy<FactorCount>>,
    Observation<FactorCount>>;
static_assert(sample::ExternallyPreparedSamplingPolicy<TerminalPolicy<2U>>);
static_assert(sample::ExternallyPreparedSamplingPolicy<CalendarPolicy<7U>>);
}  // namespace

template<std::size_t FactorCount>
void launch_rough_heston_terminal_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t parameter_count, std::size_t paths_per_parameter,
    std::uint32_t maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t dynamics_seed,
    float* device_spots
) {
    sample::launch_device_prepared_terminal_samples_cuda<
        TerminalPolicy<FactorCount>>(
        device_prepared_dynamics, parameter_count, paths_per_parameter,
        maturity_days, sample_offset, launch_sample_count, threads_per_block,
        block_count, dynamics_seed, {device_spots}, "rough_heston.samples",
        "Rough-Heston terminal sample kernel");
}

template<std::size_t FactorCount>
void launch_rough_heston_random_terminal_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t parameter_count, std::size_t paths_per_parameter,
    std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t schedule_seed,
    std::uint64_t dynamics_seed, std::uint32_t* device_maturity_days
    ,
    float* device_spots
) {
    sample::launch_device_prepared_random_terminal_samples_cuda<
        TerminalPolicy<FactorCount>>(
        device_prepared_dynamics, parameter_count, paths_per_parameter,
        {minimum_maturity_days, maximum_maturity_days}, sample_offset,
        launch_sample_count, threads_per_block, block_count, schedule_seed,
        dynamics_seed, device_maturity_days, {device_spots},
        "rough_heston.samples", "Rough-Heston random terminal sample kernel");
}

template<std::size_t FactorCount>
void launch_rough_heston_calendar_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t parameter_count, std::size_t paths_per_parameter,
    std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days,
    std::uint32_t observation_count, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t dynamics_seed,
    float* device_spots
) {
    sample::launch_device_prepared_calendar_samples_cuda<
        CalendarPolicy<FactorCount>>(
        device_prepared_dynamics, parameter_count, paths_per_parameter,
        first_observation_day, observation_interval_days, observation_count,
        sample_offset, launch_sample_count, threads_per_block, block_count,
        dynamics_seed, {device_spots}, "rough_heston.samples",
        "Rough-Heston calendar sample kernel");
}

#define AI_FACTORY_INSTANTIATE_SAMPLE(FACTORS) \
    template void launch_rough_heston_terminal_samples_cuda<FACTORS>( \
        const PreparedDynamics<FACTORS>*, std::size_t, std::size_t, \
        std::uint32_t, std::size_t, std::size_t, unsigned int, std::size_t, \
        std::uint64_t, float*); \
    template void launch_rough_heston_random_terminal_samples_cuda<FACTORS>( \
        const PreparedDynamics<FACTORS>*, std::size_t, std::size_t, \
        std::uint32_t, std::uint32_t, std::size_t, std::size_t, unsigned int, \
        std::size_t, std::uint64_t, std::uint64_t, std::uint32_t* \
        , float*); \
    template void launch_rough_heston_calendar_samples_cuda<FACTORS>( \
        const PreparedDynamics<FACTORS>*, std::size_t, std::size_t, \
        std::uint32_t, std::uint32_t, std::uint32_t, std::size_t, std::size_t, \
        unsigned int, std::size_t, std::uint64_t, float*)

AI_FACTORY_INSTANTIATE_SAMPLE(2U);
AI_FACTORY_INSTANTIATE_SAMPLE(3U);
AI_FACTORY_INSTANTIATE_SAMPLE(7U);
#undef AI_FACTORY_INSTANTIATE_SAMPLE

}  // namespace ai_factory::workbench::model::equity::rough_heston
