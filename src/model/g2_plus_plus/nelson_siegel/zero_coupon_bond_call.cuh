// Public launcher for G2++ Nelson-Siegel zero-coupon bond calls.
#pragma once

#include "curve/nelson_siegel/dataset.hpp"
#include "model/g2_plus_plus/dataset.hpp"
#include "product/zero_coupon_bond_call/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel {

// Launch one closed-form zero-coupon bond call per CUDA thread.
void launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_call_cuda(
    const G2PlusPlusModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::ZeroCouponBondCallParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::g2_plus_plus::nelson_siegel
