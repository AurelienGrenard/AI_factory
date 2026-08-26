// Block-persistent CUDA launchers for rough-Bergomi spot samples.
#pragma once

#include "model/equity/rough_bergomi/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_bergomi {

struct RoughBergomiSampleWorkspacePlan {
    std::size_t maximum_step_count;
    std::size_t history_float_count;
    std::size_t history_bytes;
    std::size_t dynamic_shared_bytes;
};

RoughBergomiSampleWorkspacePlan plan_sample_workspace(
    float maturity,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count
);

void launch_rough_bergomi_terminal_samples_cuda(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float maturity,
    float target_dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    float* device_spots
);

void launch_rough_bergomi_calendar_samples_cuda(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    std::size_t paths_per_model,
    float first_observation_time,
    float observation_interval,
    std::uint32_t observation_count,
    float target_dt,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    float* device_spots
);

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
