// OU Bermudan swaptions over the common Longstaff-Schwartz engine.
#include "model/fixed_income/ornstein_uhlenbeck/bermudan_swaption.cuh"

#include "common/fixed_income/bermudan_swaption_continuation_state.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/simulation/early_exercise_schedule.cuh"
#include "model/fixed_income/ornstein_uhlenbeck/analytics_impl.cuh"
#include "product/bermudan_swaption/pricing_policy.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck {
namespace {

using Dynamics = joint::DynamicsPolicy;
using Schedule = simulation::ExactTransitionRegularExerciseSchedule<
    Dynamics
>;
using Analytics = BermudanSwaptionAnalyticsPolicy;
using ContinuationState =
    ::ai_factory::workbench::fixed_income::OneFactorRateContinuationState<
        Dynamics
    >;

template<SwaptionSide Side>
using PricingPolicy = product::StandaloneBermudanSwaptionPricingPolicy<
    Schedule,
    Analytics,
    Side,
    ContinuationState
>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::OneFactorHermiteBasis<3U>
>;

static_assert(longstaff_schwartz::LongstaffSchwartzPolicy<
    PricingPolicy<SwaptionSide::payer>,
    Regressor
>);

}  // namespace

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_ornstein_uhlenbeck_bermudan_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const product::BermudanSwaptionParameters* host_products,
    const product::BermudanSwaptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t monte_carlo_paths_per_price,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t blocks_per_price,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
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
        time::DayFractionTimeConfiguration{time_day_fraction},
        threads_per_block,
        blocks_per_price,
        base_seed,
        device_prices,
        device_standard_errors,
        "ornstein_uhlenbeck.bermudan_swaption",
        swaption_side_name(Side),
        "OU Bermudan swaption"
    );
}

template longstaff_schwartz::LaunchResult
launch_ornstein_uhlenbeck_bermudan_swaption_cuda<SwaptionSide::payer>(
    const ModelParameters*, std::size_t,
    const product::BermudanSwaptionParameters*,
    const product::BermudanSwaptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, unsigned int, std::size_t,
    std::uint64_t, float*, float*
);
template longstaff_schwartz::LaunchResult
launch_ornstein_uhlenbeck_bermudan_swaption_cuda<SwaptionSide::receiver>(
    const ModelParameters*, std::size_t,
    const product::BermudanSwaptionParameters*,
    const product::BermudanSwaptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, unsigned int, std::size_t,
    std::uint64_t, float*, float*
);

}  // namespace ai_factory::workbench::model::fixed_income::ornstein_uhlenbeck
