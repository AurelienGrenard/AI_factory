// Generated model-only Schobel-Zhu sample launchers.
#pragma once

#include "model/equity/markovian/schobel_zhu/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::schobel_zhu {

void launch_schobel_zhu_terminal_samples_cuda(
    const ModelParameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t maturity_days,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t dynamics_seed,
    float* device_spots,
    float* device_volatilities
);

void launch_schobel_zhu_random_terminal_samples_cuda(
    const ModelParameters* device_parameters,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t schedule_seed,
    std::uint64_t dynamics_seed,
    std::uint32_t* device_maturity_days,
    float* device_spots,
    float* device_volatilities
);

void launch_schobel_zhu_calendar_samples_cuda(
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
    float* device_spots,
    float* device_volatilities
);

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
