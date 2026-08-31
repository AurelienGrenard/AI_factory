// Generated Public launcher for ${model_display} European swaptions.
#pragma once

#include "common/price_construction.cuh"

#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/${model}/parameters.hpp"
#include "product/european_swaption/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::${model} {

// Launch one regular-schedule payer or receiver per cooperative block.
template<SwaptionSide Side>
void launch_${model}_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RegularEuropeanSwaptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices,
    std::uint32_t maximum_payment_count
);

// Launch the same pricer for arbitrary schedules stored in parallel pools.
template<SwaptionSide Side>
void launch_${model}_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::ExplicitEuropeanSwaptionParameters* device_products,
    const std::uint32_t* device_payment_times_days,
    const float* device_accrual_fractions,
    std::size_t schedule_size,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices,
    std::uint32_t maximum_payment_count
);

}  // namespace ai_factory::workbench::model::fixed_income::${model}
