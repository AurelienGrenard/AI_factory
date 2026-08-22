// Closed-form Hull-White/Nelson-Siegel European swaptions.
#include "model/fixed_income/hull_white/nelson_siegel/european_swaption.cuh"

#include "common/fixed_income/european_swaption.cuh"

#include "model/fixed_income/hull_white/nelson_siegel/analytics.cu"

#include <cstddef>

namespace ai_factory::workbench::model::hull_white::nelson_siegel {
namespace {

struct EuropeanSwaptionPolicy {
    __device__ __forceinline__ HullWhiteFittedParameters prepare_model(
        const ModelParameters& model,
        const curve::nelson_siegel::NelsonSiegelParameters& curve
    ) const {
        return compose_model(model, curve);
    }

    __device__ __forceinline__ float initial_state(
        const HullWhiteFittedParameters&
    ) const {
        return 0.0f;
    }

    template<SwaptionSide Side, typename ScheduleView>
    __device__ __forceinline__ float price(
        const HullWhiteFittedParameters& model,
        float state,
        float valuation_time,
        float exercise_time,
        float fixed_rate,
        const ScheduleView& schedule
    ) const {
        return european_swaption_price<Side>(
            model, state, valuation_time, exercise_time, fixed_rate, schedule
        );
    }
};

}  // namespace

template<SwaptionSide Side>
void launch_hull_white_nelson_siegel_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::RegularEuropeanSwaptionParameters* device_products,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    fixed_income::launch_fitted_one_factor_european_swaption<
        Side, EuropeanSwaptionPolicy
    >(
        "hull_white.nelson_siegel.european_swaption",
        device_models,
        model_count,
        device_curves,
        curve_count,
        device_products,
        product::RegularEuropeanSwaptionScheduleSource{},
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count,
        device_prices
    );
}

template<SwaptionSide Side>
void launch_hull_white_nelson_siegel_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::nelson_siegel::NelsonSiegelParameters* device_curves,
    std::size_t curve_count,
    const product::ExplicitEuropeanSwaptionParameters* device_products,
    const std::uint32_t* device_payment_times,
    const float* device_accrual_fractions,
    std::size_t schedule_size,
    std::size_t product_count,
    bool cartesian_product,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    float* device_prices
) {
    fixed_income::launch_fitted_one_factor_european_swaption<
        Side, EuropeanSwaptionPolicy
    >(
        "hull_white.nelson_siegel.european_swaption",
        device_models,
        model_count,
        device_curves,
        curve_count,
        device_products,
        product::ExplicitEuropeanSwaptionScheduleSource{
            device_payment_times,
            device_accrual_fractions,
            schedule_size,
        },
        product_count,
        cartesian_product,
        result_count,
        result_offset,
        launch_result_count,
        time_day_fraction,
        threads_per_block,
        block_count,
        device_prices
    );
}

using RegularLaunchSignature = void(
    const ModelParameters*, std::size_t,
    const curve::nelson_siegel::NelsonSiegelParameters*, std::size_t,
    const product::RegularEuropeanSwaptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
using ExplicitLaunchSignature = void(
    const ModelParameters*, std::size_t,
    const curve::nelson_siegel::NelsonSiegelParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, std::size_t, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
namespace {
[[maybe_unused]] RegularLaunchSignature* launch_instantiation_0 =
    &launch_hull_white_nelson_siegel_european_swaption_cuda<SwaptionSide::payer>;
}  // namespace
namespace {
[[maybe_unused]] RegularLaunchSignature* launch_instantiation_1 =
    &launch_hull_white_nelson_siegel_european_swaption_cuda<SwaptionSide::receiver>;
}  // namespace
namespace {
[[maybe_unused]] ExplicitLaunchSignature* launch_instantiation_2 =
    &launch_hull_white_nelson_siegel_european_swaption_cuda<SwaptionSide::payer>;
}  // namespace
namespace {
[[maybe_unused]] ExplicitLaunchSignature* launch_instantiation_3 =
    &launch_hull_white_nelson_siegel_european_swaption_cuda<SwaptionSide::receiver>;
}  // namespace

}  // namespace ai_factory::workbench::model::hull_white::nelson_siegel
