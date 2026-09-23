// Compact fixed-income plan with analytical workspace cardinality.
#pragma once

#include "common/fixed_income/price_gradients/device_preparation.cuh"
#include "common/price_gradients/device_prepared_plan.hpp"

#include <algorithm>
#include <cstdint>

namespace ai_factory::workbench::fixed_income::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

template<typename ModelPreparation, typename ProductPreparation>
struct DevicePreparedPlan : pg::DevicePreparedSensitivityPlan<
    ScenarioDevicePreparation<ModelPreparation, ProductPreparation>
> {
    std::uint32_t maximum_payment_count = 0U;
    std::uint32_t maximum_exercise_count = 0U;
};

template<
    typename ModelPreparation,
    typename CurvePreparation,
    typename ProductPreparation
>
struct CurveDevicePreparedPlan : pg::CurveDevicePreparedSensitivityPlan<
    CurveScenarioDevicePreparation<
        ModelPreparation,
        CurvePreparation,
        ProductPreparation
    >
> {
    std::uint32_t maximum_payment_count = 0U;
    std::uint32_t maximum_exercise_count = 0U;
};

template<typename Plan, typename Model, typename Product>
Plan prepare_device_sensitivities(
    std::span<const Model> models,
    std::span<const Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    using Base = pg::DevicePreparedSensitivityPlan<
        typename Plan::PreparationPolicy
    >;
    using ProductAdapter =
        typename Plan::PreparationPolicy::ProductAdapter;
    Base base = pg::prepare_device_sensitivities<Base>(
        models, products, construction, time, configuration, request
    );
    std::uint32_t maximum_payment_count = 0U;
    std::uint32_t maximum_exercise_count = 0U;
    for (const auto& product : products) {
        maximum_payment_count = std::max(
            maximum_payment_count,
            ProductAdapter::payment_count(product)
        );
        if constexpr (requires { ProductAdapter::exercise_count(product); }) {
            maximum_exercise_count = std::max(
                maximum_exercise_count,
                ProductAdapter::exercise_count(product)
            );
        }
    }
    Plan result{};
    static_cast<Base&>(result) = std::move(base);
    result.maximum_payment_count = maximum_payment_count;
    result.maximum_exercise_count = maximum_exercise_count;
    return result;
}

template<
    typename Plan,
    typename Model,
    typename Curve,
    typename Product
>
Plan prepare_curve_device_sensitivities(
    std::span<const Model> models,
    std::span<const Curve> curves,
    std::span<const Product> products,
    PriceConstruction construction,
    pg::TimeConfiguration time,
    const pg::PriceGradientConfiguration& configuration,
    pg::SensitivityRequest request
) {
    using Base = pg::CurveDevicePreparedSensitivityPlan<
        typename Plan::PreparationPolicy
    >;
    using ProductAdapter =
        typename Plan::PreparationPolicy::ProductAdapter;
    Base base = pg::prepare_curve_device_sensitivities<Base>(
        models,
        curves,
        products,
        construction,
        time,
        configuration,
        request
    );
    std::uint32_t maximum_payment_count = 0U;
    std::uint32_t maximum_exercise_count = 0U;
    for (const auto& product : products) {
        maximum_payment_count = std::max(
            maximum_payment_count,
            ProductAdapter::payment_count(product)
        );
        if constexpr (requires { ProductAdapter::exercise_count(product); }) {
            maximum_exercise_count = std::max(
                maximum_exercise_count,
                ProductAdapter::exercise_count(product)
            );
        }
    }
    Plan result{};
    static_cast<Base&>(result) = std::move(base);
    result.maximum_payment_count = maximum_payment_count;
    result.maximum_exercise_count = maximum_exercise_count;
    return result;
}

}  // namespace ai_factory::workbench::fixed_income::price_gradients
