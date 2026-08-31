// Generated Public launcher for G2++ Nelson-Siegel zero-coupon bond options.
#pragma once

#include "common/price_construction.cuh"

#include "curve/nelson_siegel/parameters.hpp"
#include "common/option_side.cuh"
#include "model/fixed_income/g2_plus_plus/parameters.hpp"
#include "product/zero_coupon_bond_option/parameters.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::nelson_siegel {

// Launch closed-form zero-coupon bond-option prices across the CUDA grid.
template<OptionSide Side>
void launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::ZeroCouponBondOptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::nelson_siegel
