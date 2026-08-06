#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace ai_factory::workbench {

// Indices addressed by one flattened model/product pricing result.
struct ModelProductIndices {
    std::size_t model_index;
    std::size_t product_index;
};

// Decode a result index for either aligned rows or a Cartesian product.
// In Cartesian mode, products vary fastest within each model.
__host__ __device__ constexpr ModelProductIndices
decode_model_product_result_index(
    std::size_t result_index,
    std::size_t product_count,
    bool cartesian_product
) noexcept {
    if (cartesian_product) {
        return {
            result_index / product_count,
            result_index % product_count,
        };
    }
    return {result_index, result_index};
}

}  // namespace ai_factory::workbench
