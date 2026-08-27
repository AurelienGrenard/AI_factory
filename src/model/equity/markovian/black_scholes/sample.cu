// Black-Scholes composition over the common model-sampling engine.
#include "model/equity/markovian/black_scholes/sample.cuh"

#include "common/sample.cuh"
#include "common/simulation/schedule.cuh"

// Include the reusable dynamics so NVCC can inline every transition.
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {

using TerminalSchedule =
    simulation::ExactTransitionTerminalSchedule<DynamicsPolicy>;
using CalendarSchedule =
    simulation::ExactTransitionStubbedRegularSchedule<DynamicsPolicy>;
using Observation = sample::SpotSampleObservation<DynamicsPolicy>;
using TerminalSamplingPolicy = sample::ModelSamplingPolicy<
    TerminalSchedule,
    Observation
>;
using CalendarSamplingPolicy = sample::ModelSamplingPolicy<
    CalendarSchedule,
    Observation
>;

static_assert(sample::SamplingPolicy<TerminalSamplingPolicy>);
static_assert(sample::SamplingPolicy<CalendarSamplingPolicy>);

}  // namespace

void launch_black_scholes_terminal_samples_cuda(
    const ModelParameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t maturity_days,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t dynamics_seed,
    float* device_spots
) {
    sample::launch_device_terminal_samples_cuda<TerminalSamplingPolicy>(
        device_parameters,
        parameter_count,
        paths_per_parameter,
        maturity_days,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        dynamics_seed,
        {device_spots},
        "black_scholes.samples",
        "Black-Scholes terminal sample kernel"
    );
}

void launch_black_scholes_calendar_samples_cuda(
    const ModelParameters* device_parameters,
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
    float* device_spots
) {
    sample::launch_device_calendar_samples_cuda<CalendarSamplingPolicy>(
        device_parameters,
        parameter_count,
        paths_per_parameter,
        first_observation_day,
        observation_interval_days,
        observation_count,
        sample_offset,
        launch_sample_count,
        threads_per_block,
        block_count,
        dynamics_seed,
        {device_spots},
        "black_scholes.samples",
        "Black-Scholes calendar sample kernel"
    );
}

}  // namespace ai_factory::workbench::model::equity::black_scholes
