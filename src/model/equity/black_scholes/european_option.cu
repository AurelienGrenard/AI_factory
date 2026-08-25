// Closed-form Black-Scholes European-option composition.
#include "model/equity/black_scholes/european_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/normal_distribution.cuh"
#include "common/time_configuration.cuh"

namespace ai_factory::workbench::black_scholes {
namespace {

template<OptionSide Side>
struct EuropeanOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::EuropeanOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        float discounted_spot;
        float discounted_strike;
        float d1;
        float d2;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::EuropeanOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years = time::year_fraction(
            product.maturity,
            time_configuration
        );
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
            product.strike * expf(-model.risk_free_rate * maturity_years),
            d1,
            d1 - volatility_sqrt_maturity,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        if constexpr (Side == OptionSide::call) {
            return row.discounted_spot * normal_cdf(row.d1)
                - row.discounted_strike * normal_cdf(row.d2);
        } else {
            return row.discounted_strike * normal_cdf(-row.d2)
                - row.discounted_spot * normal_cdf(-row.d1);
        }
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
    using Pricing = EuropeanOptionClosedFormPricingPolicy<Side>;
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
        "black_scholes.european_option",
        option_side_name(Side),
        "Black-Scholes European-option kernel"
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

}  // namespace ai_factory::workbench::black_scholes
