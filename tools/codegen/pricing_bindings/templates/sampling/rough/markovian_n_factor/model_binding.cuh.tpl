// Public generated $display Markovian N-factor launch declarations for model-sample datasets.
#pragma once

#include "model/$source_folder/dynamics.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::$namespace {

template<std::size_t FactorCount>
void launch_${model_name}_terminal_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t parameter_count, std::size_t paths_per_parameter,
    std::uint32_t maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t dynamics_seed$output_declarations);

template<std::size_t FactorCount>
void launch_${model_name}_random_terminal_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t parameter_count, std::size_t paths_per_parameter,
    std::uint32_t minimum_maturity_days,
    std::uint32_t maximum_maturity_days, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t schedule_seed,
    std::uint64_t dynamics_seed, std::uint32_t* device_maturity_days$output_declarations);

template<std::size_t FactorCount>
void launch_${model_name}_calendar_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t parameter_count, std::size_t paths_per_parameter,
    std::uint32_t first_observation_day,
    std::uint32_t observation_interval_days,
    std::uint32_t observation_count, std::size_t sample_offset,
    std::size_t launch_sample_count, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t dynamics_seed$output_declarations);

}  // namespace ai_factory::workbench::$namespace
