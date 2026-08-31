// European-option payoff composed with a terminal equity schedule.
#pragma once

#include "common/equity/concepts.cuh"
#include "common/equity/path_product_monte_carlo_policy.cuh"
#include "common/equity/path_product_policy.cuh"
#include "common/simulation/schedule.cuh"
#include "common/payoff/vanilla_option.cuh"
#include "product/european_option/parameters.hpp"

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

// Schedule-independent terminal payoff used by non-Markovian path engines.
// The engine owns path construction and observation dispatch; this policy
// owns only the contract calendar, prepared payoff, handler and final value.
template<OptionSide Side>
struct EuropeanOptionPathPolicy {
    using ProductParameters = EuropeanOptionParameters;
    using Calendar = simulation::MaturityCalendar;
    static constexpr equity::ObservationCoordinate kObservationCoordinate =
        equity::ObservationCoordinate::spot;

    struct PreparedProduct {
        float strike;
        float discount;
    };

    struct Handler {
        __device__ __forceinline__ bool on_initial_value(float) {
            return true;
        }

        __device__ __forceinline__ bool on_observation(
            std::uint32_t,
            float
        ) {
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
        return product.discount * payoff::vanilla_option_payoff<Side>(
            StatePolicy::spot(terminal),
            product.strike
        );
    }
};

static_assert(std::is_trivially_copyable_v<
    EuropeanOptionPathPolicy<OptionSide::call>::PreparedProduct
>);

template<
    simulation::TerminalSchedulePolicy SchedulePolicy,
    OptionSide Side
>
requires equity::SpotDynamicsPolicy<typename SchedulePolicy::Dynamics>
using EuropeanOptionPricingPolicy = equity::PathProductMonteCarloPricingPolicy<
    SchedulePolicy,
    EuropeanOptionPathPolicy<Side>
>;

}  // namespace ai_factory::workbench::product
