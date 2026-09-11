// $model composes the common central LSM solver with frozen-exercise delta outputs.
#include "model/equity/markovian/$model/product/american_option_price_delta.cuh"
#include "common/equity/price_delta/coupled_spot_paths.cuh"
#include "common/longstaff_schwartz/basis/laguerre.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "model/equity/markovian/$model/$dynamics_header"
#include "product/american_option/continuation_state.cuh"
#include "product/american_option/price_delta_policy.cuh"

namespace ai_factory::workbench::model::equity::$model {
namespace {
using Schedule = simulation::${schedule}MaturityAlignedExerciseSchedule<$model::DynamicsPolicy>;
using Continuation = $continuation;
using FrozenPath = $frozen_path;
template<OptionSide Side>
using Policy = product::AmericanOptionPriceDeltaPolicy<Schedule, Side, Continuation, FrozenPath>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::LaguerrePolynomialTwoFactorBasis,
    longstaff_schwartz::RegressionRefinement::$regression_refinement>;
}  // namespace

template<OptionSide Side>
longstaff_schwartz::LaunchResult launch_${model}_american_option_price_delta_cuda(
    const ModelParameters* host_models, const ModelParameters* device_models,
    std::size_t model_count,
    const product::AmericanOptionParameters* host_products,
    const product::AmericanOptionParameters* device_products, std::size_t product_count,
    PriceConstruction construction, std::size_t result_count, std::size_t paths_per_price,
    $time_declaration,
    unsigned threads_per_block, std::size_t blocks_per_price, std::uint64_t base_seed,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration bump,
    float* prices, float* price_errors, float* deltas, float* delta_errors
) {
    return longstaff_schwartz::launch_longstaff_schwartz_cuda<Policy<Side>, Regressor>(
        {make_model_product_device_inputs(device_models, model_count, device_products,
                                         product_count, construction), bump, deltas, delta_errors},
        {{host_products, product_count, construction}, host_models, model_count, bump},
        result_count, paths_per_price, $time_configuration,
        threads_per_block, blocks_per_price, base_seed, prices, price_errors,
        "$model.american_option_price_delta", option_side_name(Side), "American price-delta"
    );
}

template longstaff_schwartz::LaunchResult launch_${model}_american_option_price_delta_cuda<OptionSide::call>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*, const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, $time_types, unsigned, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*);
template longstaff_schwartz::LaunchResult launch_${model}_american_option_price_delta_cuda<OptionSide::put>(
    const ModelParameters*, const ModelParameters*, std::size_t,
    const product::AmericanOptionParameters*, const product::AmericanOptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, $time_types, unsigned, std::size_t, std::uint64_t,
    ::ai_factory::workbench::equity::price_delta::SpotBumpConfiguration, float*, float*, float*, float*);

}  // namespace ai_factory::workbench::model::equity::$model
