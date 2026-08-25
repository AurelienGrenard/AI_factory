// Public launcher for the CEV Cliquet CUDA kernel.
#pragma once

#include "model/equity/cev/parameters.hpp"
#include "product/cliquet/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::cev {

// Launch the persistent Philox pricing grid on caller-owned device arrays.
void launch_cev_cliquet_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::CliquetParameters* device_products,
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

}  // namespace ai_factory::workbench::cev
