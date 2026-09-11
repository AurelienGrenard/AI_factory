// black_scholes composes the common central LSM solver with frozen-exercise delta outputs.
#include "model/equity/markovian/black_scholes/product/american_option_price_delta.cuh"
#include "common/equity/price_delta/coupled_spot_paths.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_delta_policy.cuh"

namespace ai_factory::workbench::model::equity::black_scholes {
namespace {
using Schedule = simulation::ExactTransitionMaturityAlignedExerciseSchedule<black_scholes::DynamicsPolicy>;
using Continuation = product::SpotLogMoneynessContinuationState<black_scholes::DynamicsPolicy>;
using FrozenPath = ::ai_factory::workbench::equity::price_delta::MultiplicativeFrozenExercise;
template<OptionSide Side>
using Policy = product::AmericanOptionPriceDeltaPolicy<Schedule, Side, Continuation, FrozenPath>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis,
    longstaff_schwartz::RegressionRefinement::none>;
}  // namespace

template<OptionSide Side>
longstaff_schwartz::LaunchResult launch_black_scholes_american_option_price_delta_cuda(
    const ModelParameters* host_models, const ModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products, std::size_t product_count,
    PriceConstruction construction, std::size_t result_count, std::size_t paths_per_price,
    float day_fraction,
    unsigned threads_per_block, std::size_t blocks_per_price, std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* prices, float* price_errors, float* deltas, float* delta_errors
) {
    return longstaff_schwartz::launch_longstaff_schwartz_cuda<Policy<Side>, Regressor>(
        {make_model_product_device_inputs(device_models, model_count, device_products,
                                         product_count, construction), bump, deltas, delta_errors},
        {{host_products, product_count, construction}, host_models, model_count, bump},
        result_count, paths_per_price, simulation::ExactTransitionTimeConfiguration{day_fraction},
        threads_per_block, blocks_per_price, base_seed, prices, price_errors,
        "black_scholes.american_option_price_delta", option_side_name(Side), "American price-delta"
    );
}

template longstaff_schwartz::LaunchResult launch_black_scholes_american_option_price_delta_cuda<OptionSide::call>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*, const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, unsigned, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*);
template longstaff_schwartz::LaunchResult launch_black_scholes_american_option_price_delta_cuda<OptionSide::put>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*, const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, unsigned, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*);

}  // namespace ai_factory::workbench::model::equity::black_scholes
