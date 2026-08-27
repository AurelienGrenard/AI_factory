// Model-only rough-Heston sampling over a host-prepared Markovian lift.
#pragma once

#include "model/equity/rough/rough_heston/dynamics.cuh"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_heston {

// Prepared dynamics must use dt=1/504 and an approximation horizon covering
// every requested calendar. Production bindings are instantiated for 2, 3
// and 7 factors, consistently with European-option pricing.
template<std::size_t FactorCount>
void launch_rough_heston_terminal_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t parameter_count,
    std::size_t paths_per_parameter,
    std::uint32_t maturity_days,
    std::size_t sample_offset,
    std::size_t launch_sample_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t dynamics_seed,
    float* device_spots
);

template<std::size_t FactorCount>
void launch_rough_heston_calendar_samples_cuda(
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
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
);

}  // namespace ai_factory::workbench::model::equity::rough_heston
