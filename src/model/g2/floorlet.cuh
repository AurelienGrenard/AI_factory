// Public launcher for G2 floorlet analytics.
#pragma once

#include "model/g2/dataset.hpp"
#include "product/floorlet/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::g2 {

// Launch one closed-form floorlet price per CUDA thread.
void launch_g2_floorlet_cuda(
    const G2ModelParameters* device_models,
    std::size_t model_count,
    const product::FloorletParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::g2
