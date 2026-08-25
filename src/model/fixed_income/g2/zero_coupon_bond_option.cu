// Closed-form zero-coupon-bond-option composition under G2.
#include "model/fixed_income/g2/zero_coupon_bond_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/fixed_income/bond_option_pricing_policies.cuh"
#include "common/time_configuration.cuh"

// Keep the model-specific analytical primitives visible for device inlining.
#include "model/fixed_income/g2/analytics.cu"

namespace ai_factory::workbench::model::g2 {
namespace {

template<OptionSide Side>
using ZeroCouponBondOptionPricing =
    fixed_income::StandaloneZeroCouponBondOptionClosedFormPricingPolicy<
        ModelParameters,
        Side
    >;

static_assert(closed_form::ClosedFormPricingPolicy<
    ZeroCouponBondOptionPricing<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_g2_zero_coupon_bond_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::ZeroCouponBondOptionParameters* device_products,
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
    using Pricing = ZeroCouponBondOptionPricing<Side>;
    closed_form::launch_closed_form_cuda<Pricing>(
        make_model_product_device_inputs(
            device_models,
            model_count,
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
        "g2.zero_coupon_bond_option",
        option_side_name(Side),
        "G2 zero-coupon bond option kernel"
    );
}

template void launch_g2_zero_coupon_bond_option_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::ZeroCouponBondOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_g2_zero_coupon_bond_option_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::ZeroCouponBondOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::g2
