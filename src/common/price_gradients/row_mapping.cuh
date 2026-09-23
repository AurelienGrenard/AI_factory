// Canonical model/product lookup for aligned and Cartesian price rows.
#pragma once

#include "common/price_construction.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench::price_gradients {

struct PriceRowIndices {
    std::size_t model;
    std::size_t product;
};

__host__ __device__ inline PriceRowIndices price_row_indices(
    std::size_t row,
    PriceConstruction construction,
    std::size_t product_count
) {
    if (construction == PriceConstruction::Aligned) return {row, row};
    return {row / product_count, row % product_count};
}

}  // namespace ai_factory::workbench::price_gradients
