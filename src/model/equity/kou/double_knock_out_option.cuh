// Public launcher for the Kou Double-knock-out option CUDA kernel.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/kou/dataset.hpp"
#include "product/double_knock_out_option/dataset.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::kou {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
template<OptionSide Side>
void launch_kou_double_knock_out_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::DoubleKnockOutOptionParameters* device_products,
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

}  // namespace ai_factory::workbench::kou
