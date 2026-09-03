// Public Rough-SABR Up-no-touch hybrid-FFT launcher.
#pragma once

#include "common/option_side.cuh"
#include "common/price_construction.cuh"
#include "model/equity/rough/rough_sabr/parameters.hpp"
#include "model/equity/rough/rough_sabr/volterra_fft_workspace.cuh"
#include "product/up_no_touch/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_sabr {

void launch_rough_sabr_up_no_touch_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::UpNoTouchParameters* device_products,
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
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::equity::rough_sabr
