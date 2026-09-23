// Host-visible buffers and launch metadata for device-prepared sensitivities.
#pragma once

#include "common/price_gradients/device_preparation.cuh"
#include "common/device_inputs.cuh"
#include "common/price_construction.cuh"
#include "common/price_gradients/row_mapping.cuh"
#include "common/price_gradients/sensitivity_task.cuh"
#include "common/price_gradients/time_configuration.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::monte_carlo::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;
namespace preparation =
    ::ai_factory::workbench::price_gradients::device_preparation;

template<typename Preparation>
struct DevicePreparedInputs {
    using PreparationPolicy = Preparation;
    const typename Preparation::Model* models;
    std::size_t model_capacity;
    const typename Preparation::Product* products;
    std::size_t product_capacity;
    const pg::SensitivitySpec<typename Preparation::Parameter>* sensitivities;
    std::size_t sensitivity_capacity;

    template<typename Plan>
    __device__ __forceinline__ bool make_central(
        std::size_t row,
        const Plan& plan,
        typename Preparation::Scenario& central
    ) const {
        const auto indices = pg::price_row_indices(
            row, plan.construction, plan.product_count
        );
        return Preparation::make_central(
            models[indices.model], products[indices.product], plan.time, central
        );
    }
};

template<typename Preparation>
struct CurveDevicePreparedInputs {
    using PreparationPolicy = Preparation;
    const typename Preparation::Model* models;
    std::size_t model_capacity;
    const typename Preparation::Curve* curves;
    std::size_t curve_capacity;
    const typename Preparation::Product* products;
    std::size_t product_capacity;
    const pg::SensitivitySpec<typename Preparation::Parameter>* sensitivities;
    std::size_t sensitivity_capacity;

    template<typename Plan>
    __device__ __forceinline__ bool make_central(
        std::size_t row,
        const Plan& plan,
        typename Preparation::Scenario& central
    ) const {
        const auto indices = decode_model_curve_product_result_index_32(
            static_cast<std::uint32_t>(row),
            static_cast<std::uint32_t>(plan.curve_count),
            static_cast<std::uint32_t>(plan.product_count),
            plan.construction
        );
        return Preparation::make_central(
            models[indices.model_index],
            curves[indices.curve_index],
            products[indices.product_index],
            plan.time,
            central
        );
    }
};

template<std::size_t NodeCapacity>
struct DevicePreparedStencilOutputs {
    pg::SensitivityStencil<NodeCapacity>* stencils;
    std::size_t capacity;
    preparation::Error* error;
};

struct DevicePreparedPlan {
    pg::TimeConfiguration time;
    PriceConstruction construction;
    std::size_t model_count;
    std::size_t product_count;
    std::size_t result_count;
    std::size_t sensitivity_count;
    std::size_t curve_count = 0U;
};

static_assert(std::is_trivially_copyable_v<DevicePreparedPlan>);

}  // namespace ai_factory::workbench::monte_carlo::price_gradients
