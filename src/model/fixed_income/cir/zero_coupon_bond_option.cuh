// Public launcher for CIR zero-coupon bond options.
#pragma once

#include "common/option_side.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "product/zero_coupon_bond_option/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::cir {

// Launch one closed-form zero-coupon bond option per CUDA thread.
template<OptionSide Side>
void launch_cir_zero_coupon_bond_option_cuda(
    const CirModelParameters* device_models,
    std::size_t model_count,
    const product::ZeroCouponBondOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::cir
