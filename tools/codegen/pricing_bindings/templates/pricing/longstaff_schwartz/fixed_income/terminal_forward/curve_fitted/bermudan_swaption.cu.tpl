// Generated ${model_display}/${curve_display} Bermudan swaptions over the common LSM engine.
#include "model/fixed_income/${model}/product/${curve}/bermudan_swaption.cuh"

#include "common/fixed_income/scalar_rate_continuation_state.cuh"
#include "common/longstaff_schwartz/basis/hermite.cuh"
#include "common/longstaff_schwartz/longstaff_schwartz_kernels.cuh"
#include "common/longstaff_schwartz/small_linear_regressor.cuh"
#include "common/simulation/terminal_forward_exercise_schedule.cuh"
#include "model/fixed_income/${forward_dynamics}/forward_measure_impl.cuh"
#include "model/fixed_income/${model}/${curve}/analytics_impl.cuh"
#include "product/bermudan_swaption/terminal_forward_pricing_policy.cuh"

#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::${model}::${curve} {
namespace {

using Dynamics = ${forward_dynamics}::terminal_forward::DynamicsPolicy;
using Schedule = simulation::TerminalForwardRegularExerciseSchedule<
    Dynamics
>;
using Analytics = BermudanSwaptionAnalyticsPolicy;
using ContinuationState =
    ::ai_factory::workbench::fixed_income::ScalarRateContinuationState<
        Dynamics
    >;
using CurveParameters = curve::${curve}::${curve_type}Parameters;
template<SwaptionSide Side>
using PricingPolicy = product::TerminalForwardBermudanSwaptionPricingPolicy<
    product::FittedBermudanSwaptionPricingPolicy<
        Schedule, CurveParameters, Analytics, Side, ContinuationState
    >,
    TerminalForwardBondAnalyticsPolicy
>;
using Regressor = longstaff_schwartz::NormalEquationRegressor<
    longstaff_schwartz::basis::OneFactorHermiteBasis<3U>
>;

static_assert(longstaff_schwartz::LongstaffSchwartzPolicy<
    PricingPolicy<SwaptionSide::payer>, Regressor
>);

}  // namespace

template<SwaptionSide Side>
longstaff_schwartz::LaunchResult
launch_${model}_${curve}_bermudan_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::${curve}::${curve_type}Parameters* device_curves,
    std::size_t curve_count,
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
        PricingPolicy<Side>, Regressor
    >(
        make_model_curve_product_device_inputs(
            device_models, model_count, device_curves, curve_count,
            device_products, product_count, construction
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
        "${model}.${curve}.bermudan_swaption",
        swaption_side_name(Side),
        "${model_display} ${curve_display} Bermudan swaption"
    );
}

template longstaff_schwartz::LaunchResult
launch_${model}_${curve}_bermudan_swaption_cuda<SwaptionSide::payer>(
    const ModelParameters*, std::size_t,
    const curve::${curve}::${curve_type}Parameters*, std::size_t,
    const product::BermudanSwaptionParameters*,
    const product::BermudanSwaptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, unsigned int, std::size_t,
    std::uint64_t, float*, float*
);
template longstaff_schwartz::LaunchResult
launch_${model}_${curve}_bermudan_swaption_cuda<SwaptionSide::receiver>(
    const ModelParameters*, std::size_t,
    const curve::${curve}::${curve_type}Parameters*, std::size_t,
    const product::BermudanSwaptionParameters*,
    const product::BermudanSwaptionParameters*, std::size_t,
    PriceConstruction, std::size_t, std::size_t, float, unsigned int, std::size_t,
    std::uint64_t, float*, float*
);

}  // namespace ai_factory::workbench::model::fixed_income::${model}::${curve}
