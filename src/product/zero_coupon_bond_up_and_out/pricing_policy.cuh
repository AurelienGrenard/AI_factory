// Grid-monitored zero-coupon bond barrier call under a joint rate/integral path.
#pragma once

#include "common/device_inputs.cuh"
#include "common/simulation/schedule.cuh"
#include "product/zero_coupon_bond_up_and_out/parameters.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ai_factory::workbench::product {

template<simulation::DenseSchedulePolicy SchedulePolicy,
         typename AnalyticsParameters = typename SchedulePolicy::Dynamics::Parameters>
struct ZeroCouponBondUpAndOutPricingPolicy {
    using Schedule = SchedulePolicy;
    using ModelParameters = typename Schedule::Dynamics::Parameters;
    using ProductParameters = ZeroCouponBondUpAndOutParameters;
    using TimeConfiguration = typename Schedule::TimeConfiguration;
    using DeviceInputs = ModelProductDeviceInputs<ModelParameters, ProductParameters>;

    struct HostInputs {
        const ProductParameters* products;
        std::size_t product_count;
        PriceConstruction construction;

        void validate(std::size_t result_count, const TimeConfiguration& time) const {
            simulation::validate_time_configuration(time);
            if (products == nullptr || product_count == 0U || result_count == 0U
                || (construction == PriceConstruction::Aligned && product_count != result_count))
                throw std::invalid_argument("Invalid zero-coupon bond barrier dimensions.");
            if (std::fabs(static_cast<double>(time.dt) * time.simulation_steps_per_day
                          - 1.0 / 252.0) > 1e-8)
                throw std::invalid_argument("Barrier grid must use 252 business days per year.");
            for (std::size_t i = 0U; i < product_count; ++i) {
                const auto& p = products[i];
                simulation::validate_calendar(simulation::MaturityCalendar{p.option_expiry_days}, time);
                simulation::validate_calendar(simulation::MaturityCalendar{p.bond_maturity_days}, time);
                if (!std::isfinite(p.notional) || p.notional <= 0.0f
                    || !std::isfinite(p.strike) || p.strike <= 0.0f
                    || !std::isfinite(p.barrier) || p.barrier <= 0.0f
                    || p.bond_maturity_days <= p.option_expiry_days)
                    throw std::invalid_argument("Barrier call requires N,K,B > 0 and 0 < T < U.");
            }
        }
    };

    struct PreparedRow {
        typename Schedule::PreparedSchedule simulation;
        AnalyticsParameters analytics;
        float notional;
        float strike;
        float barrier;
        float expiry_years;
        float maturity_years;
        float dt;
        std::uint32_t transition_count;
    };

    __device__ __forceinline__ static PreparedRow prepare_from_analytics(
        const ModelParameters& model,
        const AnalyticsParameters& analytics,
        const ProductParameters& product,
        const TimeConfiguration& time
    ) {
        const auto calendar = simulation::MaturityCalendar{product.option_expiry_days};
        const auto schedule = Schedule::prepare(model, calendar, time);
        return {
            schedule, analytics, product.notional, product.strike, product.barrier,
            static_cast<float>(product.option_expiry_days) / 252.0f,
            static_cast<float>(product.bond_maturity_days) / 252.0f,
            time.dt, schedule.transition_count,
        };
    }

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const ProductParameters& product,
        const TimeConfiguration& time
    ) requires std::is_same_v<AnalyticsParameters, ModelParameters> {
        return prepare_from_analytics(model, model, product, time);
    }

    struct Handler {
        AnalyticsParameters analytics;
        float barrier;
        float maturity_years;
        float expiry_years;
        float dt;
        std::uint32_t transition_count;
        float terminal_bond = 0.0f;
        bool breached = false;
        bool invalid = false;

        __device__ __forceinline__ bool observe(
            const typename Schedule::Dynamics::State& state, float time_years
        ) {
            // Unqualified call deliberately resolves the bound model analytics.
            terminal_bond = zero_coupon_bond(analytics, state.state, time_years,
                                             maturity_years);
            invalid = !isfinite(terminal_bond);
            breached = terminal_bond >= barrier;
            return !invalid && !breached;
        }

        __device__ __forceinline__ bool on_initial_state(
            const typename Schedule::Dynamics::State& state
        ) {
            return observe(state, 0.0f);
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation,
            const typename Schedule::Dynamics::State& state
        ) {
            const float time_years = observation + 1U == transition_count
                ? expiry_years : static_cast<float>(observation + 1U) * dt;
            return observe(state, time_years);
        }
    };

    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row, philox::PhiloxKey key, std::size_t path
    ) {
        Handler handler{row.analytics, row.barrier, row.maturity_years,
                        row.expiry_years, row.dt, row.transition_count};
        const auto terminal = Schedule::simulate(row.simulation, key, path, handler);
        if (handler.invalid) return nanf("");
        if (handler.breached) return 0.0f;
        const float discount = discount_factor(row.analytics, terminal.state_integral,
                                               row.expiry_years);
        if (!isfinite(discount)) return nanf("");
        return row.notional * discount * fmaxf(handler.terminal_bond - row.strike, 0.0f);
    }
};

template<simulation::DenseSchedulePolicy SchedulePolicy, typename Composition>
struct FittedZeroCouponBondUpAndOutPricingPolicy :
    ZeroCouponBondUpAndOutPricingPolicy<SchedulePolicy, typename Composition::FittedModel> {
    using Base = ZeroCouponBondUpAndOutPricingPolicy<
        SchedulePolicy, typename Composition::FittedModel>;
    using ModelParameters = typename Composition::ModelParameters;
    using CurveParameters = typename Composition::CurveParameters;
    using ProductParameters = typename Base::ProductParameters;
    using DeviceInputs = ModelCurveProductDeviceInputs<
        ModelParameters, CurveParameters, ProductParameters>;
    using typename Base::PreparedRow;
    using typename Base::TimeConfiguration;
    using typename Base::HostInputs;
    using typename Base::Schedule;

    __device__ __forceinline__ static PreparedRow prepare_row(
        const ModelParameters& model,
        const CurveParameters& curve,
        const ProductParameters& product,
        const TimeConfiguration& time
    ) {
        return Base::prepare_from_analytics(
            model, Composition::compose(model, curve), product, time);
    }
};

}  // namespace ai_factory::workbench::product
