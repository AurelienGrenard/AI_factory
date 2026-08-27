// Black-Scholes American-option composition over shared LSM kernels.
#include "model/equity/markovian/black_scholes/american_option.cuh"

#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/simulation/early_exercise_schedule.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/pricing_policy.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {

using Schedule = simulation::ExactTransitionMaturityAlignedExerciseSchedule<
    black_scholes::DynamicsPolicy
>;
using ContinuationState = product::SpotLogMoneynessContinuationState<
    black_scholes::DynamicsPolicy
>;
template<OptionSide Side>
using PricingPolicy = product::AmericanOptionPricingPolicy<
    Schedule,
    Side,
    ContinuationState
>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis
>;

static_assert(longstaff_schwartz::LongstaffSchwartzPolicy<
    PricingPolicy<OptionSide::call>,
    Regressor
>);

template<OptionSide Side>
constexpr const char* product_name() {
    if constexpr (Side == OptionSide::call) return "American-call";
    return "American-put";
}

}  // namespace

template<OptionSide Side>
longstaff_schwartz::LaunchResult launch_black_scholes_american_option_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float day_fraction,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    const simulation::ExactTransitionTimeConfiguration time_configuration{
        day_fraction,
    };
    return longstaff_schwartz::launch_longstaff_schwartz_cuda<
        PricingPolicy<Side>,
        Regressor
    >(
        make_model_product_device_inputs(
            device_models,
            model_count,
            device_products,
            product_count,
            construction
        ),
        {host_products, product_count, construction},
        result_count,
        monte_carlo_paths_per_price,
        time_configuration,
        threads_per_block,
        blocks_per_price,
        base_seed,
        device_prices,
        device_standard_errors,
        "black_scholes.american_option",
        option_side_name(Side),
        product_name<Side>()
    );
}

template longstaff_schwartz::LaunchResult
launch_black_scholes_american_option_cuda<OptionSide::call>(
    const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*,
    const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float,
    unsigned int, std::size_t, std::uint64_t, float*, float*
);
template longstaff_schwartz::LaunchResult
launch_black_scholes_american_option_cuda<OptionSide::put>(
    const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*,
    const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float,
    unsigned int, std::size_t, std::uint64_t, float*, float*
);

}  // namespace ai_factory::workbench::model::equity::black_scholes
