// Public launcher for Hull-White Svensson caplet analytics.
#pragma once

#include "curve/svensson/dataset.hpp"
#include "model/hull_white/dataset.hpp"
#include "product/caplet/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::model::hull_white::svensson {

// Launch one closed-form caplet price per CUDA thread.
void launch_hull_white_svensson_caplet_cuda(
    const HullWhiteModelParameters* device_models,
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

}  // namespace ai_factory::workbench::model::hull_white::svensson
