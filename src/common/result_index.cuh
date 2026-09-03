#pragma once

#include "common/price_construction.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench {

// Indices addressed by one flattened model/product pricing result.
struct ModelProductIndices {
    std::size_t model_index;
    std::size_t product_index;
};

// Indices addressed by one flattened model/curve/product pricing result.
struct ModelCurveProductIndices {
    std::size_t model_index;
    std::size_t curve_index;
    std::size_t product_index;
};

// Decode a result index for either aligned rows or a Cartesian product.
// In Cartesian mode, products vary fastest within each model.
__host__ __device__ constexpr ModelProductIndices
decode_model_product_result_index(
    std::size_t result_index,
    std::size_t product_count,
    PriceConstruction construction
) noexcept {
    if (is_cartesian(construction)) {
        return {
            result_index / product_count,
            result_index % product_count,
        };
    }
    return {result_index, result_index};
}

// Decode the same canonical mapping with host-validated 32-bit device indices.
// Global cardinalities and addresses remain size_t at the public boundary.
__host__ __device__ constexpr ModelProductIndices
decode_model_product_result_index_32(
    std::uint32_t result_index,
    std::uint32_t product_count,
    PriceConstruction construction
) noexcept {
    if (is_cartesian(construction)) {
        return {
            result_index / product_count,
            result_index % product_count,
        };
    }
    return {result_index, result_index};
}

// Decode aligned rows or model-major, curve-middle, product-fastest results.
__host__ __device__ constexpr ModelCurveProductIndices
decode_model_curve_product_result_index(
    std::size_t result_index,
    std::size_t curve_count,
    std::size_t product_count,
    PriceConstruction construction
) noexcept {
    if (is_cartesian(construction)) {
        const std::size_t curve_product_count = curve_count * product_count;
        const std::size_t remainder = result_index % curve_product_count;
        return {
            result_index / curve_product_count,
            remainder / product_count,
            remainder % product_count,
        };
    }
    return {result_index, result_index, result_index};
}

// 32-bit device specialization of the model/curve/product canonical mapping.
__host__ __device__ constexpr ModelCurveProductIndices
decode_model_curve_product_result_index_32(
    std::uint32_t result_index,
    std::uint32_t curve_count,
    std::uint32_t product_count,
    PriceConstruction construction
) noexcept {
    if (is_cartesian(construction)) {
        const std::uint32_t curve_product_count =
            curve_count * product_count;
        const std::uint32_t remainder = result_index % curve_product_count;
        return {
            result_index / curve_product_count,
            remainder / product_count,
            remainder % product_count,
        };
    }
    return {result_index, result_index, result_index};
}

}  // namespace ai_factory::workbench
