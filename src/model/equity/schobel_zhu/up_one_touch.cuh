// Public launcher for the Schobel-Zhu Up-one-touch CUDA kernel.
#pragma once

#include "model/equity/schobel_zhu/dataset.hpp"
#include "product/up_one_touch/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::schobel_zhu {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
void launch_schobel_zhu_up_one_touch_cuda(
    const SchobelZhuModelParameters* device_models,
    std::size_t model_count,
    const product::UpOneTouchParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float target_dt,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}  // namespace ai_factory::workbench::schobel_zhu
