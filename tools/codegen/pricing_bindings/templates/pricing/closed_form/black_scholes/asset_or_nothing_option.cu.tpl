// Generated closed-form Black-Scholes asset-or-nothing option composition.
#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {
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
            time::year_fraction(product.maturity_days, time_configuration);
        const auto values = prepare_vanilla_option_values(
            prepare_analytics(model), product.strike, maturity_years
        );
        return {
            values.discounted_underlying,
            values.d1,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        constexpr float option_sign =
            Side == OptionSide::call ? 1.0f : -1.0f;
        return asset_or_nothing_price(
            row.discounted_spot, row.d1, option_sign
        );
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
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    using PricingPolicy = AssetOrNothingOptionClosedFormPricingPolicy<Side>;
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
        "black_scholes.asset_or_nothing_option",
        option_side_name(Side),
        "Black-Scholes Asset-or-Nothing Option kernel"
    );
}

template void launch_black_scholes_asset_or_nothing_option_cuda<
    OptionSide::call
>(
    const ModelParameters*, std::size_t,
    const product::AssetOrNothingOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_black_scholes_asset_or_nothing_option_cuda<
    OptionSide::put
>(
    const ModelParameters*, std::size_t,
    const product::AssetOrNothingOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
