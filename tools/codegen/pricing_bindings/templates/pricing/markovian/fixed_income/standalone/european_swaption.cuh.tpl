// Generated exact joint-transition MC launcher for ${model_display} European swaptions.
#pragma once

#include "common/price_construction.cuh"
#include "common/fixed_income/swaption_side.cuh"
#include "model/fixed_income/${model}/parameters.hpp"
#include "product/european_swaption/parameters.hpp"
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::${model} {

template<SwaptionSide Side>
void launch_${model}_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RegularEuropeanSwaptionParameters* host_products,
    const product::RegularEuropeanSwaptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

template<SwaptionSide Side>
void launch_${model}_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::ExplicitEuropeanSwaptionParameters* host_products,
    const product::ExplicitEuropeanSwaptionParameters* device_products,
    const std::uint32_t* host_payment_times_days,
    const float* host_accrual_fractions,
    const std::uint32_t* device_payment_times_days,
    const float* device_accrual_fractions,
    std::size_t schedule_size,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::fixed_income::${model}
