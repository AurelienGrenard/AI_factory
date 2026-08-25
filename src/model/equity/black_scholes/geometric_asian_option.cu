// Closed-form Black-Scholes geometric-Asian-option composition.
#include "model/equity/black_scholes/geometric_asian_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/normal_distribution.cuh"
#include "common/simulation/schedule.cuh"

namespace ai_factory::workbench::black_scholes {
namespace {

template<OptionSide Side>
struct GeometricAsianOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::GeometricAsianOptionParameters
    >;
    using TimeConfiguration = simulation::FixedStepTimeConfiguration;

    struct PreparedRow {
        float discounted_geometric_mean;
        float discounted_strike;
        float d1;
        float d2;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::GeometricAsianOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const std::uint32_t num_steps =
            time_configuration.simulation_steps_per_day * product.maturity;
        const float maturity_years =
            static_cast<float>(num_steps) * time_configuration.dt;
        const float variance = model.volatility * model.volatility;
        const float step_count = static_cast<float>(num_steps);
        const float log_mean = logf(model.spot)
            + 0.5f * (model.risk_free_rate - model.dividend_yield
                - 0.5f * variance) * maturity_years;
        const float log_variance = variance * maturity_years
            * (2.0f * step_count + 1.0f)
            / (6.0f * (step_count + 1.0f));
        const float log_standard_deviation = sqrtf(log_variance);
        const float d2 =
            (log_mean - logf(product.strike)) / log_standard_deviation;
        return {
            expf(-model.risk_free_rate * maturity_years
                + log_mean + 0.5f * log_variance),
            product.strike * expf(-model.risk_free_rate * maturity_years),
            d2 + log_standard_deviation,
            d2,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        if constexpr (Side == OptionSide::call) {
            return row.discounted_geometric_mean * normal_cdf(row.d1)
                - row.discounted_strike * normal_cdf(row.d2);
        } else {
            return row.discounted_strike * normal_cdf(-row.d2)
                - row.discounted_geometric_mean * normal_cdf(-row.d1);
        }
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    GeometricAsianOptionClosedFormPricingPolicy<OptionSide::call>
>);

}  // namespace

template<OptionSide Side>
void launch_black_scholes_geometric_asian_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::GeometricAsianOptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float dt,
    std::uint32_t simulation_steps_per_day,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    using Pricing = GeometricAsianOptionClosedFormPricingPolicy<Side>;
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
        simulation::FixedStepTimeConfiguration{
            dt,
            simulation_steps_per_day,
        },
        threads_per_block,
        block_count,
        device_prices,
        "black_scholes.geometric_asian_option",
        option_side_name(Side),
        "Black-Scholes GeometricAsianOption kernel"
    );
}

template void launch_black_scholes_geometric_asian_option_cuda<
    OptionSide::call
>(
    const ModelParameters*, std::size_t,
    const product::GeometricAsianOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, float*
);
template void launch_black_scholes_geometric_asian_option_cuda<
    OptionSide::put
>(
    const ModelParameters*, std::size_t,
    const product::GeometricAsianOptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t,
    float, std::uint32_t, unsigned int, std::size_t, float*
);

}  // namespace ai_factory::workbench::black_scholes
