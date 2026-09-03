// Generated $display composition over the common sample engine.
#include "model/$source_folder/sample.cuh"

#include "common/sample.cuh"
#include "common/simulation/schedule.cuh"
#include "$dynamics_header"
$extra_include
namespace ai_factory::workbench::$namespace {
namespace {
using State = typename DynamicsPolicy::State;
using Observation = $observation;
using TerminalPolicy = sample::ModelSamplingPolicy<
    simulation::${schedule_prefix}TerminalSchedule<DynamicsPolicy>, Observation>;
using CalendarPolicy = sample::ModelSamplingPolicy<
    simulation::${schedule_prefix}StubbedRegularSchedule<DynamicsPolicy>, Observation>;
static_assert(sample::SamplingPolicy<TerminalPolicy>);
static_assert(sample::SamplingPolicy<CalendarPolicy>);
}  // namespace

void launch_${model_name}_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t maturity_days,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed$output_declarations
) {
    sample::launch_device_terminal_samples_cuda<TerminalPolicy>(
        device_parameters, parameter_count, paths_per_parameter, maturity_days,
        sample_offset, launch_sample_count, threads_per_block, block_count,
        dynamics_seed, {$output_values}, "$model_name.samples",
        "$display terminal sample kernel");
}

void launch_${model_name}_random_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t schedule_seed,
    std::uint64_t dynamics_seed, std::uint32_t* device_maturity_days
    $output_declarations_inline
) {
    sample::launch_device_random_terminal_samples_cuda<TerminalPolicy>(
        device_parameters, parameter_count, paths_per_parameter,
        {minimum_maturity_days, maximum_maturity_days}, sample_offset,
        launch_sample_count, threads_per_block, block_count, schedule_seed,
        dynamics_seed, device_maturity_days, {$output_values},
        "$model_name.samples", "$display random terminal sample kernel");
}

void launch_${model_name}_calendar_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days, std::uint32_t observation_count,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed$output_declarations
) {
    sample::launch_device_calendar_samples_cuda<CalendarPolicy>(
        device_parameters, parameter_count, paths_per_parameter,
        first_observation_day, observation_interval_days, observation_count,
        sample_offset, launch_sample_count, threads_per_block, block_count,
        dynamics_seed, {$output_values}, "$model_name.samples",
        "$display calendar sample kernel");
}

}  // namespace ai_factory::workbench::$namespace
