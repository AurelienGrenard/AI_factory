// Asset-or-nothing payoff composed with a terminal equity schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/option_side.cuh"
#include "common/simulation/schedule.cuh"
#include "product/asset_or_nothing_option/parameters.hpp"

#include <cstdint>

namespace ai_factory::workbench::product {

template<OptionSide Side>
struct AssetOrNothingOptionPathPolicy {
    using ProductParameters = AssetOrNothingOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        __device__ __forceinline__ bool on_initial_value(float) { return true; }
        __device__ __forceinline__ bool on_observation(std::uint32_t, float) {
            return true;
        }
    };

    __host__ __device__ static Calendar calendar(
        const ProductParameters& product
    ) {
        return {product.maturity_days};
    }

    template<typename ModelParameters>
    __device__ __forceinline__ static PreparedProduct prepare_product(
        const ModelParameters& model,
        const ProductParameters& product,
        equity::ProductPreparationContext context
    ) {
        return {
            product.strike,
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
        const Handler&
    ) {
        const float spot = StatePolicy::spot(terminal);
        const bool pays = Side == OptionSide::call
            ? spot > product.strike
            : spot < product.strike;
        return pays ? product.discount * spot : 0.0f;
    }
};

template<
    simulation::TerminalSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using AssetOrNothingOptionPricingPolicy =
    equity::PathProductMonteCarloPricingPolicy<
        SchedulePolicy,
        AssetOrNothingOptionPathPolicy<Side>
    >;

}  // namespace ai_factory::workbench::product
