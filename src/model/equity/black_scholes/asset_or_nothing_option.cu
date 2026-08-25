// Closed-form Black-Scholes asset-or-nothing option composition.
#include "model/equity/black_scholes/asset_or_nothing_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/normal_distribution.cuh"
#include "common/time_configuration.cuh"

namespace ai_factory::workbench::black_scholes {
namespace {

template<OptionSide Side>
struct AssetOrNothingOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::AssetOrNothingOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        float discounted_spot;
        float d1;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::AssetOrNothingOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years =
            time::year_fraction(product.maturity, time_configuration);
        const float sqrt_maturity = sqrtf(maturity_years);
        const float volatility_sqrt_maturity =
            model.volatility * sqrt_maturity;
        const float variance = model.volatility * model.volatility;
        const float d1 = (
            logf(model.spot / product.strike)
            + (model.risk_free_rate - model.dividend_yield
               + 0.5f * variance) * maturity_years
        ) / volatility_sqrt_maturity;
        return {
            model.spot * expf(-model.dividend_yield * maturity_years),
            d1,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        if constexpr (Side == OptionSide::call) {
            return row.discounted_spot * normal_cdf(row.d1);
        } else {
            return row.discounted_spot * normal_cdf(-row.d1);
        }
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    AssetOrNothingOptionClosedFormPricingPolicy<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_black_scholes_asset_or_nothing_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::AssetOrNothingOptionParameters* device_products,
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
    using Pricing = AssetOrNothingOptionClosedFormPricingPolicy<Side>;
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
        "black_scholes.asset_or_nothing_option",
        option_side_name(Side),
        "Black-Scholes AssetOrNothingOption kernel"
    );
}

template void launch_black_scholes_asset_or_nothing_option_cuda<
    OptionSide::call
>(
    const ModelParameters*, std::size_t,
    const product::AssetOrNothingOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_black_scholes_asset_or_nothing_option_cuda<
    OptionSide::put
>(
    const ModelParameters*, std::size_t,
    const product::AssetOrNothingOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::black_scholes
