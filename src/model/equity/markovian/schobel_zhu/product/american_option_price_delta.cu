// schobel_zhu composes the common central LSM solver with frozen-exercise delta outputs.
#include "model/equity/markovian/schobel_zhu/product/american_option_price_delta.cuh"
#include "common/equity/price_delta/coupled_spot_paths.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/equity/markovian/schobel_zhu/dynamics_impl.cuh"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_delta_policy.cuh"

namespace ai_factory::workbench::model::equity::schobel_zhu {
namespace {
using Schedule = simulation::FixedStepMaturityAlignedExerciseSchedule<schobel_zhu::DynamicsPolicy>;
using Continuation = product::SpotAndScaledStateContinuationState<schobel_zhu::DynamicsPolicy, &schobel_zhu::State::volatility, &schobel_zhu::ModelParameters::long_run_volatility>;
using FrozenPath = ::ai_factory::workbench::equity::price_delta::MultiplicativeFrozenExercise;
template<OptionSide Side>
using Policy = product::AmericanOptionPriceDeltaPolicy<Schedule, Side, Continuation, FrozenPath>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis,
    longstaff_schwartz::RegressionRefinement::none>;
}  // namespace

template<OptionSide Side>
longstaff_schwartz::LaunchResult launch_schobel_zhu_american_option_price_delta_cuda(
    const ModelParameters* host_models, const ModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products, std::size_t product_count,
    PriceConstruction construction, std::size_t result_count, std::size_t paths_per_price,
    float dt, std::uint32_t simulation_steps_per_day,
    unsigned threads_per_block, std::size_t blocks_per_price, std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* prices, float* price_errors, float* deltas, float* delta_errors
) {
    return longstaff_schwartz::launch_longstaff_schwartz_cuda<Policy<Side>, Regressor>(
        {make_model_product_device_inputs(device_models, model_count, device_products,
                                         product_count, construction), bump, deltas, delta_errors},
        {{host_products, product_count, construction}, host_models, model_count, bump},
        result_count, paths_per_price, simulation::FixedStepTimeConfiguration{dt, simulation_steps_per_day},
        threads_per_block, blocks_per_price, base_seed, prices, price_errors,
        "schobel_zhu.american_option_price_delta", option_side_name(Side), "American price-delta"
    );
}

template longstaff_schwartz::LaunchResult launch_schobel_zhu_american_option_price_delta_cuda<OptionSide::call>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*, const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, std::uint32_t, unsigned, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*);
template longstaff_schwartz::LaunchResult launch_schobel_zhu_american_option_price_delta_cuda<OptionSide::put>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*, const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, std::uint32_t, unsigned, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*);

}  // namespace ai_factory::workbench::model::equity::schobel_zhu
