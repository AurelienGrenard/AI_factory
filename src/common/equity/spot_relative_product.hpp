// Resolve spot-relative equity quotes against the unbumped model row.
#pragma once

#include "common/price_construction.cuh"
#include "common/result_index.cuh"
#include "product/range_accrual/parameters.hpp"

#include <cmath>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

namespace ai_factory::workbench::equity::spot_relative {

// Only the range-accrual barriers are already interpreted as S(t)/S(0).
// Forward-start moneyness and cliquet returns are also already relative.
template<typename Product>
constexpr bool is_quoted_coordinate(std::string_view name) {
    if constexpr (std::is_same_v<Product, product::RangeAccrualParameters>) {
        return false;
    }
    return name == "product.strike"
        || name == "product.trigger_strike"
        || name == "product.payoff_strike"
        || name == "product.barrier"
        || name == "product.lower_barrier"
        || name == "product.upper_barrier"
        || name == "product.autocall_barrier"
        || name == "product.coupon_barrier"
        || name == "product.protection_barrier";
}

template<typename Product>
inline constexpr bool kNeedsSpotResolution =
    !std::is_same_v<Product, product::RangeAccrualParameters>
    && (requires(Product value) { value.strike; }
        || requires(Product value) { value.trigger_strike; }
        || requires(Product value) { value.barrier; }
        || requires(Product value) { value.lower_barrier; }
        || requires(Product value) { value.autocall_barrier; });

template<typename Product>
Product resolve_product(const Product& quote, float reference_spot) {
    if (!std::isfinite(reference_spot) || !(reference_spot > 0.0f)) {
        throw std::invalid_argument("Equity quote requires a positive finite central spot.");
    }
    Product result = quote;
    const auto level = [reference_spot](float ratio) {
        const float absolute = ratio * reference_spot;
        if (!std::isfinite(absolute) || !(absolute > 0.0f)) {
            throw std::invalid_argument("Equity quote resolves to an invalid spot level.");
        }
        return absolute;
    };
    if constexpr (requires { result.strike; }) result.strike = level(quote.strike);
    if constexpr (requires { result.trigger_strike; })
        result.trigger_strike = level(quote.trigger_strike);
    if constexpr (requires { result.payoff_strike; })
        result.payoff_strike = level(quote.payoff_strike);
    if constexpr (requires { result.barrier; }) result.barrier = level(quote.barrier);
    if constexpr (requires { result.lower_barrier; result.upper_barrier; }) {
        if constexpr (!std::is_same_v<Product, product::RangeAccrualParameters>) {
            result.lower_barrier = level(quote.lower_barrier);
            result.upper_barrier = level(quote.upper_barrier);
        }
    }
    if constexpr (requires { result.autocall_barrier; })
        result.autocall_barrier = level(quote.autocall_barrier);
    if constexpr (requires { result.coupon_barrier; })
        result.coupon_barrier = level(quote.coupon_barrier);
    if constexpr (requires { result.protection_barrier; })
        result.protection_barrier = level(quote.protection_barrier);
    if constexpr (requires { result.reference_spot; })
        result.reference_spot = reference_spot;
    return result;
}

template<typename Model, typename Product>
struct ResolvedRows {
    std::vector<Model> models;
    std::vector<Product> products;
    PriceConstruction construction = PriceConstruction::Aligned;
};

// Distinct-S0 Cartesian rows are flattened model-major/product-fastest.
// Common-S0 and already-relative products keep the original compact layout.
// Either path retains exactly the original result-row order.
template<typename Model, typename Product>
ResolvedRows<Model, Product> resolve_rows(
    std::span<const Model> models,
    std::span<const Product> products,
    PriceConstruction construction
) {
    const auto count = price_row_count(models.size(), products.size(), construction);
    ResolvedRows<Model, Product> result;
    if (construction == PriceConstruction::CartesianProduct
        && (!kNeedsSpotResolution<Product>
            || std::all_of(models.begin(), models.end(),
                [&](const Model& model) {
                    return model.spot == models.front().spot;
                }))) {
        result.models.assign(models.begin(), models.end());
        result.products.reserve(products.size());
        for (const auto& quote : products) {
            result.products.push_back(resolve_product(quote, models.front().spot));
        }
        result.construction = PriceConstruction::CartesianProduct;
        return result;
    }
    result.models.reserve(count);
    result.products.reserve(count);
    for (std::size_t row = 0U; row < count; ++row) {
        const auto indices = decode_model_product_result_index(
            row, products.size(), construction
        );
        const auto& model = models[indices.model_index];
        result.models.push_back(model);
        result.products.push_back(resolve_product(
            products[indices.product_index], model.spot
        ));
    }
    return result;
}

template<typename Model, typename Product>
ResolvedRows<Model, Product> resolve_rows(
    const std::vector<Model>& models,
    const std::vector<Product>& products,
    PriceConstruction construction
) {
    return resolve_rows<Model, Product>(
        std::span<const Model>(models), std::span<const Product>(products),
        construction
    );
}

}  // namespace ai_factory::workbench::equity::spot_relative
