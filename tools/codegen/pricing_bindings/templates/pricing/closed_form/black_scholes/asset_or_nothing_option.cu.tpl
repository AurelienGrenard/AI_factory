// Generated closed-form Black-Scholes asset-or-nothing option composition.
#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option.cuh"
#include "model/equity/markovian/black_scholes/product/asset_or_nothing_option_impl.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/markovian/black_scholes/analytics_impl.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {

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
