// Closed-form Black-Scholes digital-option composition.
#include "model/equity/black_scholes/digital_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/time_configuration.cuh"
#include "model/equity/black_scholes/analytics.cu"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {

template<OptionSide Side>
struct DigitalOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::DigitalOptionParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        float discounted_cash_payoff;
        float d2;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::DigitalOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float maturity_years =
            time::year_fraction(product.maturity, time_configuration);
        const auto values = prepare_vanilla_option_values(
            prepare_analytics(model), product.strike, maturity_years
        );
        return {
            product.cash_payoff
                * expf(-model.risk_free_rate * maturity_years),
            values.d2,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        constexpr float option_sign =
            Side == OptionSide::call ? 1.0f : -1.0f;
        return cash_or_nothing_price(
            row.discounted_cash_payoff, row.d2, option_sign
        );
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    DigitalOptionClosedFormPricingPolicy<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_black_scholes_digital_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::DigitalOptionParameters* device_products,
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
    using PricingPolicy = DigitalOptionClosedFormPricingPolicy<Side>;
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
        "black_scholes.digital_option",
        option_side_name(Side),
        "Black-Scholes Digital Option kernel"
    );
}

template void launch_black_scholes_digital_option_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::DigitalOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
template void launch_black_scholes_digital_option_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::DigitalOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
