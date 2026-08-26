// Closed-form Black-Scholes European-option composition.
#include "model/equity/black_scholes/european_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/black_scholes/analytics.cu"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {

template<OptionSide Side>
struct EuropeanOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::EuropeanOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::EuropeanOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years = time::year_fraction(
            product.maturity,
            time_configuration
        );
        return prepare_vanilla_option_values(
            prepare_analytics(model), product.strike, maturity_years
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
    EuropeanOptionClosedFormPricingPolicy<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_black_scholes_european_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::EuropeanOptionParameters* device_products,
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
    using PricingPolicy = EuropeanOptionClosedFormPricingPolicy<Side>;
    closed_form::launch_closed_form_cuda<PricingPolicy>(
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
        "black_scholes.european_option",
        option_side_name(Side),
        "Black-Scholes European Option kernel"
    );
}

template void launch_black_scholes_european_option_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::EuropeanOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_black_scholes_european_option_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::EuropeanOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
