// Public launcher for Ornstein-Uhlenbeck caplet analytics.
#pragma once

#include "model/ornstein_uhlenbeck/dataset.hpp"
#include "product/caplet/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {

// Launch one closed-form caplet price per CUDA thread.
void launch_ornstein_uhlenbeck_caplet_cuda(
    const OrnsteinUhlenbeckModelParameters* device_models,
    std::size_t model_count,
    const product::CapletParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
);

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
