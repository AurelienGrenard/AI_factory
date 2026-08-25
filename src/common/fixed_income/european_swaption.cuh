// Shared closed-form European-swaption pricing policies and launchers.
#pragma once

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/fixed_income/swaption_side.cuh"
#include "common/time_configuration.cuh"
#include "product/european_swaption/parameters.hpp"
#include "product/european_swaption/schedule.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace ai_factory::workbench::fixed_income {

template<typename PreparedModel, typename ScheduleView>
struct PreparedEuropeanSwaptionRow {
    PreparedModel model;
    float notional;
    float strike;
    float exercise_time;
    ScheduleView schedule;
};

template<typename Model, typename Product, typename Source>
__device__ __forceinline__ auto prepare_european_swaption_row(
    const Model& model,
    const Product& product,
    Source source,
    float time_day_fraction
) {
    auto schedule = product::make_european_swaption_schedule_view(
        product,
        source,
        time_day_fraction
    );
    return PreparedEuropeanSwaptionRow<
        Model,
        decltype(schedule)
    >{
        model,
        product.notional,
        product.strike,
        static_cast<float>(product.exercise_time) * time_day_fraction,
        schedule,
    };
}

template<typename Composition, typename Model, typename Curve,
         typename Product, typename Source>
__device__ __forceinline__ auto prepare_european_swaption_row(
    const Model& model,
    const Curve& curve,
    const Product& product,
    Source source,
    float time_day_fraction
) {
    auto prepared_model = Composition::compose(model, curve);
    auto schedule = product::make_european_swaption_schedule_view(
        product,
        source,
        time_day_fraction
    );
    return PreparedEuropeanSwaptionRow<
        decltype(prepared_model),
        decltype(schedule)
    >{
        prepared_model,
        product.notional,
        product.strike,
        static_cast<float>(product.exercise_time) * time_day_fraction,
        schedule,
    };
}

template<SwaptionSide Side, typename State, typename Row>
__device__ __forceinline__ float evaluate_european_swaption_price(
    const Row& row,
    State initial_state
) {
    return row.notional * european_swaption_price<Side>(
        row.model,
        initial_state,
        0.0f,
        row.exercise_time,
        row.strike,
        row.schedule
    );
}

template<
    SwaptionSide Side,
    typename Model,
    typename Product,
    typename Source
>
struct OneFactorEuropeanSwaptionClosedFormPricingPolicy {
    using PrimaryInputs = ModelProductDeviceInputs<Model, Product>;
    using DeviceInputs = DeviceInputsWithContext<PrimaryInputs, Source>;
    using TimeConfiguration = time::DayFractionTimeConfiguration;
    using ScheduleView = decltype(
        product::make_european_swaption_schedule_view(
            std::declval<const Product&>(),
            std::declval<Source>(),
            0.0f
        )
    );
    using PreparedRow =
        PreparedEuropeanSwaptionRow<Model, ScheduleView>;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const Product& product,
        Source source,
        const TimeConfiguration& time_configuration
    ) {
        return prepare_european_swaption_row(
            model,
            product,
            source,
            time_configuration.day_fraction
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        return evaluate_european_swaption_price<Side>(
            row,
            row.model.initial_state
        );
    }
};

template<
    SwaptionSide Side,
    typename Composition,
    typename Model,
    typename Curve,
    typename Product,
    typename Source
>
struct FittedOneFactorEuropeanSwaptionClosedFormPricingPolicy {
    using PrimaryInputs =
        ModelCurveProductDeviceInputs<Model, Curve, Product>;
    using DeviceInputs = DeviceInputsWithContext<PrimaryInputs, Source>;
    using TimeConfiguration = time::DayFractionTimeConfiguration;
    using PreparedModel = decltype(
        Composition::compose(
            std::declval<const Model&>(), std::declval<const Curve&>()
        )
    );
    using ScheduleView = decltype(
        product::make_european_swaption_schedule_view(
            std::declval<const Product&>(),
            std::declval<Source>(),
            0.0f
        )
    );
    using PreparedRow =
        PreparedEuropeanSwaptionRow<PreparedModel, ScheduleView>;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const Model& model,
        const Curve& curve,
        const Product& product,
        Source source,
        const TimeConfiguration& time_configuration
    ) {
        return prepare_european_swaption_row<Composition>(
            model,
            curve,
            product,
            source,
            time_configuration.day_fraction
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        return evaluate_european_swaption_price<Side>(
            row,
            Composition::initial_state()
        );
    }
};

template<
    SwaptionSide Side,
    typename Model,
    typename Product,
    typename Source
>
void launch_one_factor_european_swaption(
    const char* kernel_name,
    const Model* device_models,
    std::size_t model_count,
    const Product* device_products,
    Source schedule_source,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    using Pricing =
        OneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side,
            Model,
            Product,
            Source
        >;
    static_assert(closed_form::ClosedFormPricingPolicy<Pricing>);
    closed_form::launch_closed_form_cuda<Pricing>(
        with_device_context(
            make_model_product_device_inputs(
                device_models,
                model_count,
                device_products,
                product_count,
                cartesian_product
            ),
            schedule_source
        ),
        result_count,
        result_offset,
        launch_result_count,
        time::DayFractionTimeConfiguration{time_day_fraction},
        threads_per_block,
        block_count,
        device_prices,
        kernel_name,
        swaption_side_name(Side),
        "European swaption kernel"
    );
}

template<
    SwaptionSide Side,
    typename Composition,
    typename Model,
    typename Curve,
    typename Product,
    typename Source
>
void launch_fitted_one_factor_european_swaption(
    const char* kernel_name,
    const Model* device_models,
    std::size_t model_count,
    const Curve* device_curves,
    std::size_t curve_count,
    const Product* device_products,
    Source schedule_source,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    using Pricing =
        FittedOneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side,
            Composition,
            Model,
            Curve,
            Product,
            Source
        >;
    static_assert(closed_form::ClosedFormPricingPolicy<Pricing>);
    closed_form::launch_closed_form_cuda<Pricing>(
        with_device_context(
            make_model_curve_product_device_inputs(
                device_models,
                model_count,
                device_curves,
                curve_count,
                device_products,
                product_count,
                cartesian_product
            ),
            schedule_source
        ),
        result_count,
        result_offset,
        launch_result_count,
        time::DayFractionTimeConfiguration{time_day_fraction},
        threads_per_block,
        block_count,
        device_prices,
        kernel_name,
        swaption_side_name(Side),
        "European swaption kernel"
    );
}

}  // namespace ai_factory::workbench::fixed_income
