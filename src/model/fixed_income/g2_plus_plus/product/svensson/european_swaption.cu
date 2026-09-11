// Generated thin composition: joint exact dynamics, swaption payoff, shared MC kernel.
#include "model/fixed_income/g2_plus_plus/product/svensson/european_swaption.cuh"
#include "model/fixed_income/g2_plus_plus/dynamics.cuh"
#include "model/fixed_income/g2_plus_plus/svensson/analytics_impl.cuh"
#include "common/monte_carlo/monte_carlo_kernel.cuh"
#include "product/european_swaption/monte_carlo_pricing_policy.cuh"

namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson {
namespace {
using Schedule = simulation::ExactTransitionTerminalSchedule<
    ::ai_factory::workbench::model::fixed_income::g2_plus_plus::joint::DynamicsPolicy>;
template<typename Product, typename Source, SwaptionSide Side>
using PricingPolicy = product::FittedEuropeanSwaptionMonteCarloPricingPolicy<
    Schedule, FittedModelComposition, Product, Source, Side>;
}  // namespace

template<SwaptionSide Side>
void launch_g2_plus_plus_svensson_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::svensson::SvenssonParameters* device_curves,
    std::size_t curve_count,
    const product::RegularEuropeanSwaptionParameters* host_products,
    const product::RegularEuropeanSwaptionParameters* device_products,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    using Pricing = PricingPolicy<product::RegularEuropeanSwaptionParameters,
        product::RegularEuropeanSwaptionScheduleSource, Side>;
    static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<Pricing>);
    const typename Pricing::DeviceInputs inputs{
        make_model_curve_product_device_inputs(
            device_models, model_count, device_curves, curve_count, device_products, product_count, construction),
        {}
    };
    monte_carlo::launch_monte_carlo_cuda<Pricing>(
        inputs, {host_products, product_count, construction, {}},
        result_count, result_offset, launch_result_count, monte_carlo_paths_per_price,
        simulation::ExactTransitionTimeConfiguration{time_day_fraction},
        threads_per_block, block_count, base_seed, device_prices, device_standard_errors,
        "g2_plus_plus.svensson.european_swaption", swaption_side_name(Side),
        "G2++ European swaption Monte Carlo kernel"
    );
}

template void launch_g2_plus_plus_svensson_european_swaption_cuda<SwaptionSide::payer>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::RegularEuropeanSwaptionParameters*,
    const product::RegularEuropeanSwaptionParameters*,
    std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t,
    std::size_t, float, unsigned int, std::size_t, std::uint64_t, float*, float*
);

template void launch_g2_plus_plus_svensson_european_swaption_cuda<SwaptionSide::receiver>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::RegularEuropeanSwaptionParameters*,
    const product::RegularEuropeanSwaptionParameters*,
    std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t,
    std::size_t, float, unsigned int, std::size_t, std::uint64_t, float*, float*
);

template<SwaptionSide Side>
void launch_g2_plus_plus_svensson_european_swaption_cuda(
    const ModelParameters* device_models,
    std::size_t model_count,
    const curve::svensson::SvenssonParameters* device_curves,
    std::size_t curve_count,
    const product::ExplicitEuropeanSwaptionParameters* host_products,
    const product::ExplicitEuropeanSwaptionParameters* device_products,
    const std::uint32_t* host_payment_times_days,
    const float* host_accrual_fractions,
    const std::uint32_t* device_payment_times_days,
    const float* device_accrual_fractions,
    std::size_t schedule_size,
    std::size_t product_count,
    PriceConstruction construction,
    std::size_t result_count,
    std::size_t result_offset,
    std::size_t launch_result_count,
    std::size_t monte_carlo_paths_per_price,
    float time_day_fraction,
    unsigned int threads_per_block,
    std::size_t block_count,
    std::uint64_t base_seed,
    float* device_prices,
    float* device_standard_errors
) {
    using Pricing = PricingPolicy<product::ExplicitEuropeanSwaptionParameters,
        product::ExplicitEuropeanSwaptionScheduleSource, Side>;
    static_assert(monte_carlo::ScalarMonteCarloPricingPolicy<Pricing>);
    const typename Pricing::DeviceInputs inputs{
        make_model_curve_product_device_inputs(
            device_models, model_count, device_curves, curve_count, device_products, product_count, construction),
        {device_payment_times_days, device_accrual_fractions, schedule_size, product_count}
    };
    monte_carlo::launch_monte_carlo_cuda<Pricing>(
        inputs, {host_products, product_count, construction, {host_payment_times_days, host_accrual_fractions, schedule_size, product_count}},
        result_count, result_offset, launch_result_count, monte_carlo_paths_per_price,
        simulation::ExactTransitionTimeConfiguration{time_day_fraction},
        threads_per_block, block_count, base_seed, device_prices, device_standard_errors,
        "g2_plus_plus.svensson.european_swaption", swaption_side_name(Side),
        "G2++ European swaption Monte Carlo kernel"
    );
}

template void launch_g2_plus_plus_svensson_european_swaption_cuda<SwaptionSide::payer>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, const std::uint32_t*, const float*, std::size_t,
    std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t,
    std::size_t, float, unsigned int, std::size_t, std::uint64_t, float*, float*
);

template void launch_g2_plus_plus_svensson_european_swaption_cuda<SwaptionSide::receiver>(
    const ModelParameters*, std::size_t,
    const curve::svensson::SvenssonParameters*, std::size_t,
    const product::ExplicitEuropeanSwaptionParameters*,
    const product::ExplicitEuropeanSwaptionParameters*,
    const std::uint32_t*, const float*, const std::uint32_t*, const float*, std::size_t,
    std::size_t, PriceConstruction, std::size_t, std::size_t, std::size_t,
    std::size_t, float, unsigned int, std::size_t, std::uint64_t, float*, float*
);

}  // namespace ai_factory::workbench::model::fixed_income::g2_plus_plus::svensson
