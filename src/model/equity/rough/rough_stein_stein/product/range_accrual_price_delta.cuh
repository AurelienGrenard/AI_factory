// Public Rough Stein-Stein Range-accrual hybrid-FFT launcher.
#pragma once

#include "common/option_side.cuh"
#include "common/price_construction.cuh"
#include "model/equity/rough/rough_stein_stein/parameters.hpp"
#include "common/equity/price_delta/spot_bump.cuh"
#include "common/volterra/hybrid_fft_price_delta_workspace.cuh"
#include "product/range_accrual/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_stein_stein {

void launch_rough_stein_stein_range_accrual_price_delta_cuda(
    const ModelParameters* host_models,
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* host_products,
    const product::RangeAccrualParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_index,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    float target_dt,
    std::size_t step_count,
    std::size_t path_chunk_size,
    void* device_workspace,
    std::size_t workspace_bytes,
    std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* device_prices,
    float* device_standard_errors,
    float* device_deltas,
    float* device_delta_errors
);

}  // namespace ai_factory::workbench::model::equity::rough_stein_stein
