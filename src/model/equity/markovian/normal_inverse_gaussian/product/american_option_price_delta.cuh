// normal_inverse_gaussian American price and frozen-exercise spot delta; separate from price-only.
#pragma once

#include "common/equity/price_delta/spot_bump.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "common/option_side.cuh"
#include "common/price_construction.cuh"
#include "model/equity/markovian/normal_inverse_gaussian/parameters.hpp"
#include "product/american_option/parameters.hpp"
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::normal_inverse_gaussian {

template<OptionSide Side>
longstaff_schwartz::LaunchResult launch_normal_inverse_gaussian_american_option_price_delta_cuda(
    const ModelParameters* host_models, const ModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products, std::size_t product_count,
    PriceConstruction construction, std::size_t result_count, std::size_t paths_per_price,
    float day_fraction,
    unsigned threads_per_block, std::size_t blocks_per_price, std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* prices, float* price_errors, float* deltas, float* delta_errors
);

}  // namespace ai_factory::workbench::model::equity::normal_inverse_gaussian
