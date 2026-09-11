// Model/product/calendar preparation shared by analytic and Monte Carlo swaptions.
#pragma once

#include "product/european_swaption/schedule.cuh"

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

}  // namespace ai_factory::workbench::fixed_income
