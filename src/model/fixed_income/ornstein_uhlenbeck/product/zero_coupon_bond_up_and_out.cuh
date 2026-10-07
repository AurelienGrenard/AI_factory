// Generated public OU zero-coupon bond up-and-out call launcher.
#pragma once

#include "common/price_construction.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/parameters.hpp"
#include "product/zero_coupon_bond_up_and_out/parameters.hpp"
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {

void launch_ornstein_uhlenbeck_zero_coupon_bond_up_and_out_cuda(
    const ModelParameters* device_models, std::size_t model_count,
    const product::ZeroCouponBondUpAndOutParameters* host_products,
    const product::ZeroCouponBondUpAndOutParameters* device_products,
    std::size_t product_count, PriceConstruction construction,
    std::size_t result_count, std::size_t result_offset,
    std::size_t launch_result_count, std::size_t paths_per_price,
    std::uint32_t simulation_steps_per_day, unsigned int threads_per_block,
    std::size_t block_count, std::uint64_t base_seed,
    float* device_prices, float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
