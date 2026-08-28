// Public Rough-Heston Asian-option N-factor launcher.
#pragma once

#include "common/option_side.cuh"
#include "common/price_construction.cuh"
#include "model/equity/rough/rough_heston/dynamics.cuh"
#include "product/asian_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_heston {

template<OptionSide Side, std::size_t FactorCount>
void launch_rough_heston_asian_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const PreparedDynamics<FactorCount>* device_prepared_dynamics,
    std::size_t prepared_dynamics_count,
    const product::AsianOptionParameters* device_products,
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
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::equity::rough_heston
