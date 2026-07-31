// Public launcher for Hull-White Nelson-Siegel caplet analytics.
#pragma once

#include "curve/nelson_siegel/dataset.hpp"
#include "model/hull_white/dataset.hpp"
#include "product/caplet/dataset.hpp"

#include <cstddef>

namespace ai_factory::workbench::hull_white::nelson_siegel {

// Launch one closed-form caplet price per CUDA thread.
void launch_hull_white_nelson_siegel_caplet_cuda(
    const HullWhiteModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
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

}  // namespace ai_factory::workbench::hull_white::nelson_siegel
