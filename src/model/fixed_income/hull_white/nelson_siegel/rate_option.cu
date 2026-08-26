// Closed-form Hull-White rate-option composition fitted to a parametric curve.
#include "model/fixed_income/hull_white/nelson_siegel/rate_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/fixed_income/bond_option_pricing_policies.cuh"
#include "common/time_configuration.cuh"

// Keep the model-specific analytical primitives visible for device inlining.
#include "model/fixed_income/hull_white/nelson_siegel/analytics.cu"

namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel {
namespace {

template<OptionSide Side>
using PricingPolicy =
    ::ai_factory::workbench::fixed_income::FittedRateOptionClosedFormPricingPolicy<
        FittedModelComposition,
        Side
    >;

static_assert(closed_form::ClosedFormPricingPolicy<
    PricingPolicy<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_hull_white_nelson_siegel_rate_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const FittedModelComposition::CurveParameters* device_curves,
    std::size_t curve_count,
    const product::RateOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    closed_form::launch_closed_form_cuda<PricingPolicy<Side>>(
        make_model_curve_product_device_inputs(
            device_models,
            model_count,
            device_curves,
            curve_count,
            device_products,
            product_count,
            cartesian_product
        ),
        result_count,
        result_offset,
        launch_result_count,
        time::DayFractionTimeConfiguration{day_fraction},
        threads_per_block,
        block_count,
        device_prices,
        "hull_white.nelson_siegel.rate_option",
        option_side_name(Side),
        "Hull-White rate_option kernel"
    );
}

template void launch_hull_white_nelson_siegel_rate_option_cuda<
    OptionSide::call
>(
    const ModelParameters*, std::size_t,
    const FittedModelComposition::CurveParameters*, std::size_t,
    const product::RateOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_hull_white_nelson_siegel_rate_option_cuda<
    OptionSide::put
>(
    const ModelParameters*, std::size_t,
    const FittedModelComposition::CurveParameters*, std::size_t,
    const product::RateOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel
