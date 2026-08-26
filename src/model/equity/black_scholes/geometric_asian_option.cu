// Closed-form Black-Scholes geometric-Asian-option composition.
#include "model/equity/black_scholes/geometric_asian_option.cuh"

#include "common/closed_form/closed_form_kernels.cuh"
#include "common/device_inputs.cuh"
#include "common/simulation/schedule.cuh"
#include "model/equity/black_scholes/analytics.cu"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {

template<OptionSide Side>
struct GeometricAsianOptionClosedFormPricingPolicy {
    using DeviceInputs = ModelProductDeviceInputs<
        ModelParameters,
        product::GeometricAsianOptionParameters
    >;
    using TimeConfiguration = simulation::FixedStepTimeConfiguration;

    using PreparedRow = DiscountedLognormalOptionValues;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const product::GeometricAsianOptionParameters& product,
        const TimeConfiguration& time_configuration
    ) {
        const std::uint32_t transition_count =
            time_configuration.simulation_steps_per_day * product.maturity;
        const float maturity_years =
            static_cast<float>(transition_count) * time_configuration.dt;
        return prepare_geometric_asian_option_values(
            prepare_analytics(model),
            product.strike,
            maturity_years,
            transition_count
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
    using PricingPolicy = GeometricAsianOptionClosedFormPricingPolicy<Side>;
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
        simulation::FixedStepTimeConfiguration{
            dt,
            simulation_steps_per_day,
        },
        threads_per_block,
        block_count,
        device_prices,
        "black_scholes.geometric_asian_option",
        option_side_name(Side),
        "Black-Scholes Geometric Asian Option kernel"
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

}  // namespace ai_factory::workbench::model::equity::black_scholes
