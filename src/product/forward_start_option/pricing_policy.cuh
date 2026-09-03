// Two-date forward-start payoff composed with an exact calendar schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "product/forward_start_option/parameters.hpp"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::product {

template<
    OptionSide Side,
    bool SeparateCallRounding = false
>
struct ForwardStartOptionPathPolicy {
    using ProductParameters = ForwardStartOptionParameters;
    using Calendar = simulation::StaticCalendar<2U>;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float moneyness;
        float discount;
    };

    struct Handler {
        float reset_spot = 0.0f;

        __device__ __forceinline__ bool on_initial_value(float) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation,
            float spot
        ) {
            if (observation == 0U) reset_spot = spot;
            return true;
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {{product.reset_time_days, product.maturity_days - product.reset_time_days}};
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        return {
            product.moneyness,
            expf(-model.risk_free_rate * context.maturity_years),
        };
    }

    __device__ __forceinline__ static Handler make_handler(
        const PreparedProduct&
    ) {
        return {};
    }

    template<typename StatePolicy>
    __device__ __forceinline__ static float finalize(
        const PreparedProduct& product,
        const typename StatePolicy::State& terminal,
        const Handler& handler
    ) {
        const float terminal_spot = StatePolicy::spot(terminal);
        if constexpr (Side == OptionSide::call) {
            const float reset_strike = SeparateCallRounding
                ? __fmul_rn(product.moneyness, handler.reset_spot)
                : product.moneyness * handler.reset_spot;
            const float intrinsic = SeparateCallRounding
                ? __fsub_rn(terminal_spot, reset_strike)
                : terminal_spot - reset_strike;
            return product.discount * fmaxf(intrinsic, 0.0f);
        } else {
            return product.discount * fmaxf(
                product.moneyness * handler.reset_spot - terminal_spot,
                0.0f
            );
        }
    }
};

template<
    simulation::TwoDateSchedulePolicy SchedulePolicy,
    OptionSide Side,
    bool SeparateCallRounding = false
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using ForwardStartOptionPricingPolicy =
    equity::PathProductMonteCarloPricingPolicy<
        SchedulePolicy,
        ForwardStartOptionPathPolicy<Side, SeparateCallRounding>
    >;

}  // namespace ai_factory::workbench::product
