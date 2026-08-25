// Public launcher for the {model_display} {header_product_comment} CUDA kernel.
#pragma once

#include "common/option_side.cuh"
#include "model/equity/{model}/parameters.hpp"
#include "product/{product}/parameters.hpp"

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::{model} {{

// Launch the persistent Philox pricing grid on caller-owned device arrays.
template<OptionSide Side>
void launch_{model}_{product}_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::{product_type}Parameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
{time_parameter_declarations}    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
);

}}  // namespace ai_factory::workbench::{model}
