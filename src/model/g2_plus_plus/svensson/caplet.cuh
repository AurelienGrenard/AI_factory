// Public launcher for G2++ Svensson caplet analytics.
#pragma once

#include "curve/svensson/dataset.hpp"
#include "model/g2_plus_plus/dataset.hpp"
#include "product/caplet/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::g2_plus_plus::svensson {

// Launch one closed-form caplet price per CUDA thread.
void launch_g2_plus_plus_svensson_caplet_cuda(
    const G2PlusPlusModelParameters* device_models,
    std::size_t model_count,
    const curve::svensson::SvenssonParameters* device_curves,
    std::size_t curve_count,
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

}  // namespace ai_factory::workbench::model::g2_plus_plus::svensson
