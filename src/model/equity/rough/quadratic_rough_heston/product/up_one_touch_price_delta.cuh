// Public Quadratic rough-Heston Up-one-touch N-factor price/spot-delta launcher.
#pragma once

#include "common/option_side.cuh"
#include "common/equity/price_delta/spot_bump.cuh"
#include "common/price_construction.cuh"
#include "model/equity/rough/quadratic_rough_heston/dynamics.cuh"
#include "product/up_one_touch/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

template<std::size_t FactorCount>
void launch_quadratic_rough_heston_up_one_touch_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::UpOneTouchParameters* host_products,
    const product::UpOneTouchParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_standard_errors
);

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
