// Terminal European-swaption payoff over exact joint factor/integral simulation.
// Standalone and curve-fitted models share the calendar, payoff and MC kernel.
#pragma once

#include "common/device_inputs.cuh"
#include "common/fixed_income/swaption_side.cuh"
#include "common/simulation/schedule.cuh"
#include "product/european_swaption/pricing_row.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace ai_factory::workbench::product {

template<typename Product, typename Source>
struct EuropeanSwaptionMonteCarloHostInputs {
    const Product* products;
    std::size_t product_count;
    PriceConstruction construction;
    Source schedules;

    void validate(std::size_t result_count, const simulation::ExactTransitionTimeConfiguration& time) const {
        simulation::validate_time_configuration(time);
        if (products == nullptr || product_count == 0U || result_count == 0U
            || (construction == PriceConstruction::Aligned && product_count != result_count))
            throw std::invalid_argument("Invalid European-swaption host product dimensions.");
        for (std::size_t index = 0U; index < product_count; ++index) {
            const auto& product = products[index];
            simulation::validate_calendar(simulation::MaturityCalendar{product.exercise_time_days}, time);
            if (!std::isfinite(product.notional) || product.notional <= 0.0f
                || !std::isfinite(product.strike) || product.strike < 0.0f || product.payment_count == 0U)
                throw std::invalid_argument("Invalid European-swaption payoff parameters.");
            if constexpr (std::is_same_v<Product, RegularEuropeanSwaptionParameters>) {
                const auto maturity = static_cast<std::uint64_t>(product.exercise_time_days)
                    + static_cast<std::uint64_t>(product.payment_count) * product.payment_interval_days;
                if (product.payment_interval_days == 0U || !std::isfinite(product.accrual_fraction)
                    || product.accrual_fraction <= 0.0f || maturity > std::numeric_limits<std::uint32_t>::max())
                    throw std::invalid_argument("Invalid regular European-swaption payment calendar.");
                simulation::validate_calendar(simulation::MaturityCalendar{static_cast<std::uint32_t>(maturity)}, time);
            } else {
                if (schedules.payment_times_days == nullptr || schedules.accrual_fractions == nullptr
                    || schedules.schedule_stride == 0U || product.schedule_offset >= schedules.schedule_size
                    || product.payment_count - 1U > (schedules.schedule_size - 1U - product.schedule_offset)
                        / schedules.schedule_stride)
                    throw std::invalid_argument("Invalid explicit European-swaption host schedule pools.");
                auto previous = product.exercise_time_days;
                for (std::uint32_t payment = 0U; payment < product.payment_count; ++payment) {
                    const auto offset = product.schedule_offset + static_cast<std::size_t>(payment) * schedules.schedule_stride;
                    const auto date = schedules.payment_times_days[offset];
                    const float accrual = schedules.accrual_fractions[offset];
                    if (date <= previous || !std::isfinite(accrual) || accrual <= 0.0f)
                        throw std::invalid_argument("Explicit payments must increase after exercise with positive accruals.");
                    simulation::validate_calendar(simulation::MaturityCalendar{date}, time);
                    previous = date;
                }
            }
        }
    }
};

template<typename SchedulePolicy, typename PreparedModel, typename Product,
         typename Source, SwaptionSide Side, typename PrimaryInputs>
struct EuropeanSwaptionMonteCarloPricingPolicyCore {
    using Schedule = SchedulePolicy;
    using ProductParameters = Product;
    using TimeConfiguration = typename Schedule::TimeConfiguration;
    using DeviceInputs = DeviceInputsWithContext<PrimaryInputs, Source>;
    using HostInputs = EuropeanSwaptionMonteCarloHostInputs<Product, Source>;
    using ScheduleView = decltype(make_european_swaption_schedule_view(
        std::declval<const Product&>(), std::declval<Source>(), 0.0f));
    struct PreparedRow {
        typename Schedule::PreparedSchedule simulation;
        fixed_income::PreparedEuropeanSwaptionRow<PreparedModel, ScheduleView> product;
    };

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row, philox::PhiloxKey key, std::size_t path
    ) {
        const auto terminal = Schedule::simulate_terminal(row.simulation, key, path);
        const auto& product = row.product;
        if (!product.schedule.valid()) return nanf("");
        // ADL selects the canonical model analytics, including a fitted curve shift.
        const float swap = payer_swap_value(product.model, terminal.state,
            product.exercise_time_years, product.exercise_time_years,
            product.strike, product.schedule);
        const float discount = discount_factor(product.model, terminal.state_integral,
            product.exercise_time_years);
        constexpr float sign = Side == SwaptionSide::payer ? 1.0f : -1.0f;
        // Do not let fmaxf turn a failed analytic evaluation into a valid zero.
        return isfinite(swap) && isfinite(discount)
            ? product.notional * discount * fmaxf(sign * swap, 0.0f) : nanf("");
    }
};

template<typename Schedule, typename Product, typename Source, SwaptionSide Side>
struct StandaloneEuropeanSwaptionMonteCarloPricingPolicy
    : EuropeanSwaptionMonteCarloPricingPolicyCore<Schedule, typename Schedule::Dynamics::Parameters,
        Product, Source, Side, ModelProductDeviceInputs<typename Schedule::Dynamics::Parameters, Product>> {
    using Model = typename Schedule::Dynamics::Parameters;
    using Base = EuropeanSwaptionMonteCarloPricingPolicyCore<Schedule, Model, Product, Source, Side,
        ModelProductDeviceInputs<Model, Product>>;
    __device__ __forceinline__ static typename Base::PreparedRow prepare_row(
        const Model& model, const Product& product, Source source,
        const typename Schedule::TimeConfiguration& time
    ) {
        return {Schedule::prepare(model, {product.exercise_time_days}, time),
            fixed_income::prepare_european_swaption_row(model, product, source, time.day_fraction)};
    }
};

template<typename Schedule, typename Composition, typename Product, typename Source, SwaptionSide Side>
struct FittedEuropeanSwaptionMonteCarloPricingPolicy
    : EuropeanSwaptionMonteCarloPricingPolicyCore<Schedule, typename Composition::FittedModel,
        Product, Source, Side, ModelCurveProductDeviceInputs<typename Composition::ModelParameters,
            typename Composition::CurveParameters, Product>> {
    using Base = EuropeanSwaptionMonteCarloPricingPolicyCore<Schedule, typename Composition::FittedModel,
        Product, Source, Side, ModelCurveProductDeviceInputs<typename Composition::ModelParameters,
            typename Composition::CurveParameters, Product>>;
    __device__ __forceinline__ static typename Base::PreparedRow prepare_row(
        const typename Composition::ModelParameters& model, const typename Composition::CurveParameters& curve,
        const Product& product, Source source, const typename Schedule::TimeConfiguration& time
    ) {
        return {Schedule::prepare(model, {product.exercise_time_days}, time),
            fixed_income::prepare_european_swaption_row<Composition>(model, curve, product, source, time.day_fraction)};
    }
};

}  // namespace ai_factory::workbench::product
