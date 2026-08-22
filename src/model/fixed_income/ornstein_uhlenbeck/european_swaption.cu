// Closed-form European swaptions under the Ornstein-Uhlenbeck short rate.
#include "model/fixed_income/ornstein_uhlenbeck/european_swaption.cuh"

#include "common/fixed_income/european_swaption.cuh"

// Include analytics so NVCC can inline the Jamshidian decomposition.
#include "model/fixed_income/ornstein_uhlenbeck/analytics.cu"

#include <cstddef>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {
namespace {

struct EuropeanSwaptionPolicy {
    __device__ __forceinline__ ModelParameters prepare_model(
        const ModelParameters& model
    ) const {
        return model;
    }

    __device__ __forceinline__ float initial_state(
        const ModelParameters& model
    ) const {
        return model.initial_state;
    }

    template<SwaptionSide Side, typename ScheduleView>
    __device__ __forceinline__ float price(
        const ModelParameters& model,
        float state,
        float valuation_time,
        float exercise_time,
        float fixed_rate,
        const ScheduleView& schedule
    ) const {
        return european_swaption_price<Side>(
            model,
            state,
            valuation_time,
            exercise_time,
            fixed_rate,
            schedule
        );
    }
};

}  // namespace

template<SwaptionSide Side>
void launch_ornstein_uhlenbeck_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
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
    fixed_income::launch_one_factor_european_swaption<
        Side, EuropeanSwaptionPolicy
    >(
        "ornstein_uhlenbeck.european_swaption",
        device_models,
        model_count,
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
void launch_ornstein_uhlenbeck_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
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
    fixed_income::launch_one_factor_european_swaption<
        Side, EuropeanSwaptionPolicy
    >(
        "ornstein_uhlenbeck.european_swaption",
        device_models,
        model_count,
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
    const product::RegularEuropeanSwaptionParameters*, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
using ExplicitLaunchSignature = void(
    const ModelParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, std::size_t, std::size_t,
    bool, std::size_t, std::size_t, std::size_t, float,
    unsigned int, std::size_t, float*
);
namespace {
[[maybe_unused]] RegularLaunchSignature* launch_instantiation_0 =
    &launch_ornstein_uhlenbeck_european_swaption_cuda<SwaptionSide::payer>;
}  // namespace
namespace {
[[maybe_unused]] RegularLaunchSignature* launch_instantiation_1 =
    &launch_ornstein_uhlenbeck_european_swaption_cuda<SwaptionSide::receiver>;
}  // namespace
namespace {
[[maybe_unused]] ExplicitLaunchSignature* launch_instantiation_2 =
    &launch_ornstein_uhlenbeck_european_swaption_cuda<SwaptionSide::payer>;
}  // namespace
namespace {
[[maybe_unused]] ExplicitLaunchSignature* launch_instantiation_3 =
    &launch_ornstein_uhlenbeck_european_swaption_cuda<SwaptionSide::receiver>;
}  // namespace

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
