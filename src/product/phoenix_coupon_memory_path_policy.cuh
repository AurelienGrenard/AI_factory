// Shared Phoenix path policy with compile-time selection of missed-coupon memory.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"

#include <cstdint>

namespace ai_factory::workbench::product::detail {

template<bool RememberMissedCoupons>
struct PhoenixCouponMemoryObservationHandler {
    std::uint32_t observation_count;
    float autocall_barrier;
    float coupon_barrier;
    float protection_barrier;
    float coupon_per_observation;
    float discount_per_observation;
    float present_value = 0.0f;
    float discount = 1.0f;
    float remembered_coupon = 0.0f;

    __device__ __forceinline__ bool on_initial_value(float) {
        return true;
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        float spot
    ) {
        discount *= discount_per_observation;
        float coupon = 0.0f;
        if constexpr (RememberMissedCoupons) {
            remembered_coupon += coupon_per_observation;
            coupon = spot >= coupon_barrier ? remembered_coupon : 0.0f;
            if (coupon > 0.0f) remembered_coupon = 0.0f;
        } else {
            coupon = spot >= coupon_barrier ? coupon_per_observation : 0.0f;
        }

        const bool maturity = observation + 1U == observation_count;
        if (!maturity && spot >= autocall_barrier) {
            present_value += discount * (1.0f + coupon);
            return false;
        }
        if (maturity) {
            const float capital = spot >= protection_barrier ? 1.0f : spot;
            present_value += discount * (capital + coupon);
            return false;
        }
        present_value = fmaf(discount, coupon, present_value);
        return true;
    }
};

template<typename ProductParametersT, bool RememberMissedCoupons>
struct PhoenixCouponMemoryPathPolicy {
    using ProductParameters = ProductParametersT;
    using Calendar = simulation::RegularCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        std::uint32_t observation_count;
        float autocall_barrier;
        float coupon_barrier;
        float protection_barrier;
        float coupon_per_observation;
        float discount_per_observation;
    };

    using Handler =
        PhoenixCouponMemoryObservationHandler<RememberMissedCoupons>;

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {
            product.observation_interval_days,
            product.maturity_days / product.observation_interval_days,
        };
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        const std::uint32_t observation_count =
            product.maturity_days / product.observation_interval_days;
        const float observation_years =
            static_cast<float>(product.observation_interval_days)
            * context.day_fraction;
        return {
            observation_count,
            product.autocall_barrier,
            product.coupon_barrier,
            product.protection_barrier,
            product.annual_coupon_rate * observation_years,
            expf(-model.risk_free_rate * observation_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct& product
    ) {
        return {
            product.observation_count,
            product.autocall_barrier,
            product.coupon_barrier,
            product.protection_barrier,
            product.coupon_per_observation,
            product.discount_per_observation,
        };
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct&,
        const typename StatePolicy::State&,
        const Handler& handler
    ) {
        return handler.present_value;
    }
};

template<
    simulation::CountedObservedSchedulePolicy SchedulePolicy,
    typename ProductParametersT,
    bool RememberMissedCoupons
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using PhoenixCouponMemoryPricingPolicy =
    equity::PathProductMonteCarloPricingPolicy<
        SchedulePolicy,
        PhoenixCouponMemoryPathPolicy<
            ProductParametersT,
            RememberMissedCoupons
        >
    >;

}  // namespace ai_factory::workbench::product::detail
