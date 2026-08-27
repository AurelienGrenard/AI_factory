// Public Longstaff-Schwartz launcher for G2++/NS Bermudan swaptions.
#pragma once

#include "common/price_construction.cuh"

#include "common/fixed_income/swaption_side.cuh"
#include "common/longstaff_schwartz/launch.cuh"
#include "curve/nelson_siegel/dataset.hpp"
#include "model/fixed_income/g2_plus_plus/parameters.hpp"
#include "product/bermudan_swaption/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::nelson_siegel {

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_g2_plus_plus_nelson_siegel_bermudan_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
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

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::nelson_siegel
