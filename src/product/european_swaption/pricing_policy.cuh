// Product-owned closed-form European-swaption policies and launchers.
#pragma once

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/closed_form/cooperative_closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/fixed_income/jamshidian_cooperative.cuh"
#include "common/fixed_income/swaption_side.cuh"
#include "common/time_configuration.cuh"
#include "product/european_swaption/parameters.hpp"
#include "product/european_swaption/schedule.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ai_factory::workbench::fixed_income {

template<typename PreparedModel, typename ScheduleView>
struct PreparedEuropeanSwaptionRow {
    PreparedModel model;
    float notional;
    float strike;
    float exercise_time_years;
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
        static_cast<float>(product.exercise_time_days) * time_day_fraction,
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
        static_cast<float>(product.exercise_time_days) * time_day_fraction,
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
        row.exercise_time_years,
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
    typename Provider,
    typename Model,
    typename Product,
    typename Source
>
struct CooperativeOneFactorEuropeanSwaptionClosedFormPricingPolicy
    : OneFactorEuropeanSwaptionClosedFormPricingPolicy<
          Side,
          Model,
          Product,
          Source
      > {
    using Base = OneFactorEuropeanSwaptionClosedFormPricingPolicy<
        Side,
        Model,
        Product,
        Source
    >;
    using typename Base::DeviceInputs;
    using typename Base::PreparedRow;
    using typename Base::TimeConfiguration;
    using Base::evaluate_price;
    using Base::prepare_row;

    inline static std::size_t required_shared_memory_bytes(
        std::uint32_t maximum_payment_count
    ) {
        return cooperative_jamshidian_shared_memory_bytes(
            maximum_payment_count
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row,
        std::byte* workspace,
        std::uint32_t workspace_capacity
    ) {
        return row.notional * cooperative_european_swaption_price<Side>(
            Provider{},
            row.model,
            row.model.initial_state,
            0.0f,
            row.exercise_time_years,
            row.strike,
            row.schedule,
            workspace,
            workspace_capacity
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
    typename Provider,
    typename Composition,
    typename Model,
    typename Curve,
    typename Product,
    typename Source
>
struct CooperativeFittedOneFactorEuropeanSwaptionClosedFormPricingPolicy
    : FittedOneFactorEuropeanSwaptionClosedFormPricingPolicy<
          Side,
          Composition,
          Model,
          Curve,
          Product,
          Source
      > {
    using Base = FittedOneFactorEuropeanSwaptionClosedFormPricingPolicy<
        Side,
        Composition,
        Model,
        Curve,
        Product,
        Source
    >;
    using typename Base::DeviceInputs;
    using typename Base::PreparedRow;
    using typename Base::TimeConfiguration;
    using Base::evaluate_price;
    using Base::prepare_row;

    inline static std::size_t required_shared_memory_bytes(
        std::uint32_t maximum_payment_count
    ) {
        return cooperative_jamshidian_shared_memory_bytes(
            maximum_payment_count
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row,
        std::byte* workspace,
        std::uint32_t workspace_capacity
    ) {
        return row.notional * cooperative_european_swaption_price<Side>(
            Provider{},
            row.model,
            Composition::initial_state(),
            0.0f,
            row.exercise_time_years,
            row.strike,
            row.schedule,
            workspace,
            workspace_capacity
        );
    }
};

inline std::size_t scalar_closed_form_block_count(
    std::size_t launch_result_count,
    unsigned int threads_per_block,
    std::size_t cooperative_block_count
) {
    // Let the scalar launcher report invalid empty launches or block sizes.
    // Avoid unsigned underflow and division by zero while computing its grid.
    if (launch_result_count == 0U || threads_per_block == 0U) {
        return cooperative_block_count;
    }
    const std::size_t direct_block_count =
        (launch_result_count - 1U) / threads_per_block + 1U;
    return std::min(cooperative_block_count, direct_block_count);
}

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
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    using PricingPolicy =
        OneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side,
            Model,
            Product,
            Source
        >;
    static_assert(closed_form::ClosedFormPricingPolicy<PricingPolicy>);
    closed_form::launch_closed_form_cuda<PricingPolicy>(
        with_device_context(
            make_model_product_device_inputs(
                device_models,
                model_count,
                device_products,
                product_count,
                construction
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
    typename Provider,
    typename Model,
    typename Product,
    typename Source
>
void launch_cooperative_one_factor_european_swaption(
    const char* kernel_name,
    const Model* device_models,
    std::size_t model_count,
    const Product* device_products,
    Source schedule_source,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices,
    std::uint32_t maximum_payment_count
) {
    if (maximum_payment_count == 0U) {
        throw std::invalid_argument(
            "maximum_payment_count must be positive."
        );
    }
    using PricingPolicy =
        CooperativeOneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side,
            Provider,
            Model,
            Product,
            Source
        >;
    static_assert(closed_form::ClosedFormPricingPolicy<PricingPolicy>);
    static_assert(
        closed_form::CooperativeClosedFormPricingPolicy<PricingPolicy>
    );
    const auto inputs = with_device_context(
        make_model_product_device_inputs(
            device_models,
            model_count,
            device_products,
            product_count,
            construction
        ),
        schedule_source
    );
    const time::DayFractionTimeConfiguration time_configuration{
        time_day_fraction
    };
    const bool launched = maximum_payment_count > 1U
        && closed_form::launch_cooperative_closed_form_cuda<PricingPolicy>(
            inputs,
            result_count,
            result_offset,
            launch_result_count,
            time_configuration,
            maximum_payment_count,
            threads_per_block,
            block_count,
            device_prices,
            kernel_name,
            swaption_side_name(Side),
            "Cooperative European swaption kernel"
        );
    if (launched) return;

    closed_form::launch_closed_form_cuda<PricingPolicy>(
        inputs,
        result_count,
        result_offset,
        launch_result_count,
        time_configuration,
        threads_per_block,
        scalar_closed_form_block_count(
            launch_result_count,
            threads_per_block,
            block_count
        ),
        device_prices,
        kernel_name,
        swaption_side_name(Side),
        "European swaption fallback kernel"
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
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    using PricingPolicy =
        FittedOneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side,
            Composition,
            Model,
            Curve,
            Product,
            Source
        >;
    static_assert(closed_form::ClosedFormPricingPolicy<PricingPolicy>);
    closed_form::launch_closed_form_cuda<PricingPolicy>(
        with_device_context(
            make_model_curve_product_device_inputs(
                device_models,
                model_count,
                device_curves,
                curve_count,
                device_products,
                product_count,
                construction
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
    typename Provider,
    typename Composition,
    typename Model,
    typename Curve,
    typename Product,
    typename Source
>
void launch_cooperative_fitted_one_factor_european_swaption(
    const char* kernel_name,
    const Model* device_models,
    std::size_t model_count,
    const Curve* device_curves,
    std::size_t curve_count,
    const Product* device_products,
    Source schedule_source,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices,
    std::uint32_t maximum_payment_count
) {
    if (maximum_payment_count == 0U) {
        throw std::invalid_argument(
            "maximum_payment_count must be positive."
        );
    }
    using PricingPolicy =
        CooperativeFittedOneFactorEuropeanSwaptionClosedFormPricingPolicy<
            Side,
            Provider,
            Composition,
            Model,
            Curve,
            Product,
            Source
        >;
    static_assert(closed_form::ClosedFormPricingPolicy<PricingPolicy>);
    static_assert(
        closed_form::CooperativeClosedFormPricingPolicy<PricingPolicy>
    );
    const auto inputs = with_device_context(
        make_model_curve_product_device_inputs(
            device_models,
            model_count,
            device_curves,
            curve_count,
            device_products,
            product_count,
            construction
        ),
        schedule_source
    );
    const time::DayFractionTimeConfiguration time_configuration{
        time_day_fraction
    };
    const bool launched = maximum_payment_count > 1U
        && closed_form::launch_cooperative_closed_form_cuda<PricingPolicy>(
            inputs,
            result_count,
            result_offset,
            launch_result_count,
            time_configuration,
            maximum_payment_count,
            threads_per_block,
            block_count,
            device_prices,
            kernel_name,
            swaption_side_name(Side),
            "Cooperative European swaption kernel"
        );
    if (launched) return;

    closed_form::launch_closed_form_cuda<PricingPolicy>(
        inputs,
        result_count,
        result_offset,
        launch_result_count,
        time_configuration,
        threads_per_block,
        scalar_closed_form_block_count(
            launch_result_count,
            threads_per_block,
            block_count
        ),
        device_prices,
        kernel_name,
        swaption_side_name(Side),
        "European swaption fallback kernel"
    );
}

}  // namespace ai_factory::workbench::fixed_income
