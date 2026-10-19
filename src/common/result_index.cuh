// Decode aligned and Cartesian result rows into their model, curve and product indices.
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

// One reproducible curve draw with replacement per model-product result row.
// The same key is used by prices, gradients, host validation and JSON output.
inline constexpr std::uint64_t kSampledCurveSeed = 20261014ULL;

__host__ __device__ constexpr std::size_t sampled_curve_index(
    std::size_t result_index, std::size_t curve_count
) noexcept {
    std::uint64_t value = kSampledCurveSeed
        + static_cast<std::uint64_t>(result_index)
        + 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    value ^= value >> 31U;
    return static_cast<std::size_t>(value % curve_count);
}

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
    if (construction == PriceConstruction::PairedModelCurveCartesianProduct) {
        const auto model_index = result_index / product_count;
        return {model_index, model_index, result_index % product_count};
    }
    if (construction == PriceConstruction::SampledCurveCartesianProduct) {
        return {
            result_index / product_count,
            sampled_curve_index(result_index, curve_count),
            result_index % product_count,
        };
    }
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
    if (construction == PriceConstruction::PairedModelCurveCartesianProduct) {
        const auto model_index = result_index / product_count;
        return {model_index, model_index, result_index % product_count};
    }
    if (construction == PriceConstruction::SampledCurveCartesianProduct) {
        return {
            result_index / product_count,
            sampled_curve_index(result_index, curve_count),
            result_index % product_count,
        };
    }
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
