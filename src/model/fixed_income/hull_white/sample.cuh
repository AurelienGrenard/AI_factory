// Model-only Hull-White sample launchers on the 252-day contract clock.
#pragma once
#include "model/fixed_income/hull_white/parameters.hpp"
#include <cstddef>
#include <cstdint>
namespace ai_factory::workbench::model::fixed_income::hull_white {
void launch_hull_white_terminal_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t maturity_days,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed, float* device_states
);
void launch_hull_white_calendar_samples_cuda(
    const ModelParameters* device_parameters, std::size_t parameter_count,
    std::size_t paths_per_parameter, std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days, std::uint32_t observation_count,
    std::size_t sample_offset, std::size_t launch_sample_count,
    unsigned int threads_per_block, std::size_t block_count,
    std::uint64_t dynamics_seed, float* device_states
);
}  // namespace ai_factory::workbench::model::fixed_income::hull_white
