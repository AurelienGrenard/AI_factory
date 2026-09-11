// Generated Public Longstaff-Schwartz launcher for ${model_display}/${curve_display} Bermudan swaptions.
#pragma once

#include "common/price_construction.cuh"

#include "common/fixed_income/swaption_side.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "curve/${curve}/dataset.hpp"
#include "model/fixed_income/${model}/parameters.hpp"
#include "product/bermudan_swaption/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::${model}::${curve} {

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_${model}_${curve}_bermudan_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::${curve}::${curve_type}Parameters* device_curves,
    std::size_t curve_count,
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

}  // namespace ai_factory::workbench::model::fixed_income::${model}::${curve}
