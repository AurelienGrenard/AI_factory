// Canonical host/device contract for aligned and Cartesian pricing rows.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench {

enum class PriceConstruction : std::uint8_t {
    Aligned,
    CartesianProduct,
};

__host__ __device__ constexpr bool is_cartesian(
    PriceConstruction construction
) noexcept {
    return construction == PriceConstruction::CartesianProduct;
}

// Validate one model/product construction and return its result cardinality.
inline std::size_t price_row_count(
    std::size_t model_count,
    std::size_t product_count,
    PriceConstruction construction
) {
    if (model_count == 0U || product_count == 0U) {
        throw std::invalid_argument(
            "Price construction requires non-empty model and product datasets."
        );
    }
    if (construction == PriceConstruction::Aligned) {
        if (model_count != product_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal model and product counts."
            );
        }
        return model_count;
    }
    if (model_count > std::numeric_limits<std::size_t>::max() / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    return model_count * product_count;
}

// Validate one model/curve/product construction and return its cardinality.
inline std::size_t price_row_count(
    std::size_t model_count,
    std::size_t curve_count,
    std::size_t product_count,
    PriceConstruction construction
) {
    if (model_count == 0U || curve_count == 0U || product_count == 0U) {
        throw std::invalid_argument(
            "Price construction requires non-empty input datasets."
        );
    }
    if (construction == PriceConstruction::Aligned) {
        if (model_count != curve_count || model_count != product_count) {
            throw std::invalid_argument(
                "Aligned construction requires equal input row counts."
            );
        }
        return model_count;
    }
    const std::size_t maximum = std::numeric_limits<std::size_t>::max();
    if (model_count > maximum / curve_count
        || model_count * curve_count > maximum / product_count) {
        throw std::overflow_error("Cartesian result count exceeds size_t.");
    }
    return model_count * curve_count * product_count;
}

}  // namespace ai_factory::workbench
