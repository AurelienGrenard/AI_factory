// Closed-form Hull-White/Svensson European swaptions.
#include "model/fixed_income/hull_white/svensson/european_swaption.cuh"

#include "product/european_swaption/pricing_policy.cuh"

#include "model/fixed_income/hull_white/svensson/analytics_impl.cuh"

#include <cstddef>

namespace ai_factory::workbench::model::fixed_income::hull_white::svensson {

template<SwaptionSide Side>
void launch_hull_white_svensson_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::svensson::SvenssonParameters* device_curves,
    std::size_t curve_count,
    const product::RegularEuropeanSwaptionParameters* device_products,
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
    ::ai_factory::workbench::fixed_income::launch_fitted_one_factor_european_swaption<
        Side, FittedModelComposition
    >(
        "hull_white.svensson.european_swaption",
        device_models,
        model_count,
        device_curves,
        curve_count,
        device_products,
        product::RegularEuropeanSwaptionScheduleSource{},
        product_count,
        construction,
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count,
        device_prices
    );
}

template<SwaptionSide Side>
void launch_hull_white_svensson_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::svensson::SvenssonParameters* device_curves,
    std::size_t curve_count,
    const product::ExplicitEuropeanSwaptionParameters* device_products,
    const std::uint32_t* device_payment_times_days,
    const float* device_accrual_fractions,
    std::size_t schedule_size,
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
    ::ai_factory::workbench::fixed_income::launch_cooperative_fitted_one_factor_european_swaption<
        Side, FittedAnalyticsProvider, FittedModelComposition
    >(
        "hull_white.svensson.european_swaption",
        device_models,
        model_count,
        device_curves,
        curve_count,
        device_products,
        product::ExplicitEuropeanSwaptionScheduleSource{
            device_payment_times_days,
            device_accrual_fractions,
            schedule_size,
            product_count,
        },
        product_count,
        construction,
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count,
        device_prices,
        maximum_payment_count
    );
}

template void launch_hull_white_svensson_european_swaption_cuda<
    SwaptionSide::payer
>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::RegularEuropeanSwaptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_hull_white_svensson_european_swaption_cuda<
    SwaptionSide::receiver
>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::RegularEuropeanSwaptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_hull_white_svensson_european_swaption_cuda<
    SwaptionSide::payer
>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, std::size_t, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*, std::uint32_t
);
template void launch_hull_white_svensson_european_swaption_cuda<
    SwaptionSide::receiver
>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, std::size_t, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*, std::uint32_t
);

}  // namespace ai_factory::workbench::model::fixed_income::hull_white::svensson
