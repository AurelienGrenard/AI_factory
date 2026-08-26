// Public launcher for Hull-White Svensson European swaptions.
#pragma once

#include "common/fixed_income/swaption_side.cuh"
#include "curve/svensson/parameters.hpp"
#include "model/fixed_income/hull_white/parameters.hpp"
#include "product/european_swaption/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::fixed_income::hull_white::svensson {

// Launch one regular-schedule payer (call) or receiver (put) per thread.
template<SwaptionSide Side>
void launch_hull_white_svensson_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::svensson::SvenssonParameters* device_curves,
    std::size_t curve_count,
    const product::RegularEuropeanSwaptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

// Launch the same pricer for arbitrary schedules stored in parallel pools.
template<SwaptionSide Side>
void launch_hull_white_svensson_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::svensson::SvenssonParameters* device_curves,
    std::size_t curve_count,
    const product::ExplicitEuropeanSwaptionParameters* device_products,
    const std::uint32_t* device_payment_times,
    const float* device_accrual_fractions,
    std::size_t schedule_size,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::fixed_income::hull_white::svensson
