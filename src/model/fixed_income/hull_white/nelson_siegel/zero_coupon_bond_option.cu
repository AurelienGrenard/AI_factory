// Closed-form Hull-White zero-coupon-bond-option composition.
#include "model/fixed_income/hull_white/nelson_siegel/zero_coupon_bond_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "product/zero_coupon_bond_option/pricing_policy.cuh"
#include "common/time_configuration.cuh"

// Keep the model-specific analytical primitives visible for device inlining.
#include "model/fixed_income/hull_white/nelson_siegel/analytics_impl.cuh"

namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel {
namespace {

template<OptionSide Side>
using PricingPolicy =
    ::ai_factory::workbench::fixed_income::FittedZeroCouponBondOptionClosedFormPricingPolicy<
        FittedModelComposition,
        Side
    >;

static_assert(closed_form::ClosedFormPricingPolicy<
    PricingPolicy<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_hull_white_nelson_siegel_zero_coupon_bond_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const FittedModelComposition::CurveParameters* device_curves,
    std::size_t curve_count,
    const product::ZeroCouponBondOptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
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
            construction
        ),
        result_count,
        result_offset,
        launch_result_count,
        time::DayFractionTimeConfiguration{day_fraction},
        threads_per_block,
        block_count,
        device_prices,
        "hull_white.nelson_siegel.zero_coupon_bond_option",
        option_side_name(Side),
        "Hull-White zero-coupon bond option kernel"
    );
}

template void launch_hull_white_nelson_siegel_zero_coupon_bond_option_cuda<
    OptionSide::call
>(
    const ModelParameters*, std::size_t,
    const FittedModelComposition::CurveParameters*, std::size_t,
    const product::ZeroCouponBondOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_hull_white_nelson_siegel_zero_coupon_bond_option_cuda<
    OptionSide::put
>(
    const ModelParameters*, std::size_t,
    const FittedModelComposition::CurveParameters*, std::size_t,
    const product::ZeroCouponBondOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::fixed_income::hull_white::nelson_siegel
