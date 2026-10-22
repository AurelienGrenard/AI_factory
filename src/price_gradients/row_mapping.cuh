// Canonical model/product lookup for aligned and Cartesian price rows.
#pragma once

#include "pricing_rows/result_index.cuh"

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
    const auto selected = decode_model_product_result_index(
        row, product_count, construction);
    return {selected.model_index, selected.product_index};
}

}  // namespace ai_factory::workbench::price_gradients
