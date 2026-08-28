// Public Longstaff-Schwartz launcher for Vasicek Bermudan swaptions.
#pragma once

#include "common/price_construction.cuh"

#include "common/fixed_income/swaption_side.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "model/fixed_income/vasicek/parameters.hpp"
#include "product/bermudan_swaption/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::vasicek {

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult launch_vasicek_bermudan_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::BermudanSwaptionParameters* host_products,
    const product::BermudanSwaptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::fixed_income::vasicek
