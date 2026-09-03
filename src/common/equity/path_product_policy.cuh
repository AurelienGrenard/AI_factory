// Model-independent equity path-product contract and state observation bridge.
#pragma once

#include "common/equity/concepts.cuh"

#include <concepts>
#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::equity {

enum class ObservationCoordinate : std::uint8_t {
    spot,
    log_spot,
};

// Contractual time information needed while preparing path-dependent payoffs.
// day_fraction remains separate from the numerical transition step.
struct ProductPreparationContext {
    float day_fraction;
    float maturity_years;
};

template<typename StatePolicy, ObservationCoordinate Coordinate>
concept StatePolicyForObservationCoordinate =
    (Coordinate == ObservationCoordinate::spot
        && SpotStatePolicy<StatePolicy>)
    || (Coordinate == ObservationCoordinate::log_spot
        && LogSpotStatePolicy<StatePolicy>);

template<typename ProductHandler>
concept PathProductObservationHandler =
    std::is_trivially_copyable_v<ProductHandler>
    && requires(
        ProductHandler& handler,
        std::uint32_t observation,
        float value
    ) {
        {
            handler.on_initial_value(value)
        } -> std::same_as<bool>;
        {
            handler.on_observation(observation, value)
        } -> std::same_as<bool>;
    };

template<
    typename StatePolicy,
    PathProductObservationHandler ProductHandler,
    ObservationCoordinate Coordinate
>
requires StatePolicyForObservationCoordinate<StatePolicy, Coordinate>
struct PathProductObservationAdapter {
    ProductHandler& handler;

    __device__ __forceinline__ static float coordinate(
        const typename StatePolicy::State& state
    ) {
        if constexpr (Coordinate == ObservationCoordinate::spot) {
            return StatePolicy::spot(state);
        } else {
            return StatePolicy::log_spot(state);
        }
    }

    __device__ __forceinline__ bool on_initial_state(
        const typename StatePolicy::State& state
    ) {
        return handler.on_initial_value(coordinate(state));
    }

    __device__ __forceinline__ bool on_observation(
        std::uint32_t observation,
        const typename StatePolicy::State& state
    ) {
        return handler.on_observation(observation, coordinate(state));
    }
};

template<typename ProductPolicy, typename StatePolicy>
concept EquityPathProductPolicyShape =
    requires {
        typename ProductPolicy::ProductParameters;
        typename ProductPolicy::Calendar;
        typename ProductPolicy::PreparedProduct;
        typename ProductPolicy::Handler;
        requires std::same_as<
            std::remove_cv_t<decltype(ProductPolicy::kObservationCoordinate)>,
            ObservationCoordinate
        >;
    }
    && std::is_trivially_copyable_v<typename ProductPolicy::ProductParameters>
    && std::is_trivially_copyable_v<typename ProductPolicy::Calendar>
    && std::is_trivially_copyable_v<typename ProductPolicy::PreparedProduct>
    && PathProductObservationHandler<typename ProductPolicy::Handler>;

template<typename ProductPolicy, typename StatePolicy>
concept EquityPathProductPolicy =
    EquityPathProductPolicyShape<ProductPolicy, StatePolicy>
    && StatePolicyForObservationCoordinate<
        StatePolicy,
        ProductPolicy::kObservationCoordinate
    >
    && requires(
        const typename StatePolicy::Parameters& model,
        const typename ProductPolicy::ProductParameters& parameters,
        const typename ProductPolicy::PreparedProduct& prepared,
        const typename StatePolicy::State& terminal,
        const typename ProductPolicy::Handler& const_handler,
        ProductPreparationContext context
    ) {
        {
            ProductPolicy::calendar(parameters)
        } -> std::same_as<typename ProductPolicy::Calendar>;
        {
            ProductPolicy::prepare_product(model, parameters, context)
        } -> std::same_as<typename ProductPolicy::PreparedProduct>;
        {
            ProductPolicy::make_handler(prepared)
        } -> std::same_as<typename ProductPolicy::Handler>;
        {
            ProductPolicy::template finalize<StatePolicy>(
                prepared,
                terminal,
                const_handler
            )
        } -> std::same_as<float>;
    };

}  // namespace ai_factory::workbench::equity
