// Closed-form Black-Scholes range-accrual composition.
#include "model/equity/black_scholes/range_accrual.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/normal_distribution.cuh"
#include "common/time_configuration.cuh"

namespace ai_factory::workbench::black_scholes {
namespace {

struct RangeAccrualClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::RangeAccrualParameters
    >;
    using TimeConfiguration = time::DayFractionTimeConfiguration;

    struct PreparedRow {
        float log_lower_barrier;
        float log_upper_barrier;
        float log_drift_rate;
        float volatility;
        float observation_interval;
        float maturity_discount;
        float discounted_coupon_per_observation;
        std::uint32_t observation_count;
    };

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::RangeAccrualParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const float variance = model.volatility * model.volatility;
        const float observation_years = time::year_fraction(
            product.observation_interval,
            time_configuration
        );
        const float maturity_years = time::year_fraction(
            product.maturity,
            time_configuration
        );
        const float maturity_discount =
            expf(-model.risk_free_rate * maturity_years);
        return {
            logf(product.lower_barrier),
            logf(product.upper_barrier),
            model.risk_free_rate - model.dividend_yield - 0.5f * variance,
            model.volatility,
            observation_years,
            maturity_discount,
            maturity_discount * product.coupon_rate * observation_years,
            product.maturity / product.observation_interval,
        };
    }

    __device__ __forceinline__ static float evaluate_price(
        const PreparedRow& row
    ) {
        double probability_sum = 0.0;
        for (std::uint32_t observation = 1U;
             observation <= row.observation_count;
             ++observation) {
            const float observation_time =
                static_cast<float>(observation) * row.observation_interval;
            const float standard_deviation =
                row.volatility * sqrtf(observation_time);
            const float mean = row.log_drift_rate * observation_time;
            const float lower =
                (row.log_lower_barrier - mean) / standard_deviation;
            const float upper =
                (row.log_upper_barrier - mean) / standard_deviation;
            probability_sum += static_cast<double>(
                normal_cdf(upper) - normal_cdf(lower)
            );
        }
        return fmaf(
            row.discounted_coupon_per_observation,
            static_cast<float>(probability_sum),
            row.maturity_discount
        );
    }
};

static_assert(closed_form::ClosedFormPricingPolicy<
    RangeAccrualClosedFormPricingPolicy
>);

}  // namespace

void launch_black_scholes_range_accrual_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::RangeAccrualParameters* device_products,
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
    using Pricing = RangeAccrualClosedFormPricingPolicy;
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
        "black_scholes.range_accrual",
        "none",
        "Black-Scholes Range-accrual kernel"
    );
}

}  // namespace ai_factory::workbench::black_scholes
