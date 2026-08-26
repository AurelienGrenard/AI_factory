// Public launcher for the Schobel-Zhu Down-and-in option CUDA kernel.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/schobel_zhu/parameters.hpp"
#include "product/down_and_in_option/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::schobel_zhu {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
template<OptionSide Side>
void launch_schobel_zhu_down_and_in_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::DownAndInOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float dt,
    std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
