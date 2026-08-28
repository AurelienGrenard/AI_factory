// Public Log-modulated rough-Bergomi Asset-or-nothing-option hybrid-FFT launcher.
#pragma once

#include "common/option_side.cuh"
#include "common/price_construction.cuh"
#include "model/equity/rough/log_modulated_rough_bergomi/parameters.hpp"
#include "model/equity/rough/log_modulated_rough_bergomi/volterra_fft_workspace.cuh"
#include "product/asset_or_nothing_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi {

template<OptionSide Side>
void launch_log_modulated_rough_bergomi_asset_or_nothing_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::AssetOrNothingOptionParameters* device_products,
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

}  // namespace ai_factory::workbench::model::equity::log_modulated_rough_bergomi
