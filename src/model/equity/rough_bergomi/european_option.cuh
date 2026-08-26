// Public workspace planner and launcher for rough-Bergomi European options.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/rough_bergomi/parameters.hpp"
#include "product/european_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_bergomi {

// Caller-owned storage required by one persistent rough-Bergomi launch.
struct RoughBergomiWorkspacePlan {
    std::size_t maximum_step_count;
    std::size_t history_float_count;
    std::size_t history_bytes;
    std::size_t dynamic_shared_bytes;
};

// Size the history lanes and shared hybrid grid from host product rows.
RoughBergomiWorkspacePlan plan_european_option_workspace(
    const product::EuropeanOptionParameters* host_products,
    std::size_t product_count,
    float day_fraction,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count
);

// Launch the persistent Philox pricing grid on caller-owned device arrays.
template<OptionSide Side>
void launch_rough_bergomi_european_option_cuda(
    const RoughBergomiModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::size_t maximum_step_count,
    float* device_history_workspace,
    std::size_t history_workspace_float_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::equity::rough_bergomi
