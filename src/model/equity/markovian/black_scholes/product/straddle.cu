// Generated closed-form Black-Scholes straddle composition.
#include "model/equity/markovian/black_scholes/product/straddle.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {

struct StraddleClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::StraddleParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::StraddleParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years = time::year_fraction(
            product.maturity_days,
            time_configuration
        );
        return prepare_vanilla_option_values(
            prepare_analytics(model), product.strike, maturity_years
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        return discounted_lognormal_option_price(row, 1.0f)
            + discounted_lognormal_option_price(row, -1.0f);
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    StraddleClosedFormPricingPolicy
>);

}  // namespace

void launch_black_scholes_straddle_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::StraddleParameters* device_products,
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
    using PricingPolicy = StraddleClosedFormPricingPolicy;
    closed_form::launch_closed_form_cuda<PricingPolicy>(
        make_model_product_device_inputs(
            device_models,
            model_count,
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
        "black_scholes.straddle",
        "none",
        "Black-Scholes Straddle kernel"
    );
}

}  // namespace ai_factory::workbench::model::equity::black_scholes
