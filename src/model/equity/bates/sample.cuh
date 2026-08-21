// Persistent CUDA launchers for bates model samples.
#pragma once

#include "model/equity/bates/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::bates {

void launch_bates_terminal_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float maturity,
    float dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_spots,
    float* device_variances
);

void launch_bates_calendar_samples_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    float dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_spots,
    float* device_variances
);

}  // namespace ai_factory::workbench::bates
