// Generated closed-form Black-Scholes gap-option composition.
#include "model/equity/markovian/black_scholes/gap_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {

template<OptionSide Side>
struct GapOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::GapOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::GapOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years = time::year_fraction(
            product.maturity_days,
            time_configuration
        );
        return prepare_gap_option_values(
            prepare_analytics(model),
            product.trigger_strike,
            product.payoff_strike,
            maturity_years
        );
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        constexpr float option_sign =
            Side == OptionSide::call ? 1.0f : -1.0f;
        return discounted_lognormal_option_price(row, option_sign);
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    GapOptionClosedFormPricingPolicy<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_black_scholes_gap_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::GapOptionParameters* device_products,
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
    using PricingPolicy = GapOptionClosedFormPricingPolicy<Side>;
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
        "black_scholes.gap_option",
        option_side_name(Side),
        "Black-Scholes Gap Option kernel"
    );
}

template void launch_black_scholes_gap_option_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::GapOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_black_scholes_gap_option_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::GapOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
