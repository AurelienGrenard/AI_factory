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

template<
    LogSpotStatePolicy StatePolicy,
    typename ProductHandler,
    ObservationCoordinate Coordinate
>
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
concept EquityPathProductPolicy =
    LogSpotStatePolicy<StatePolicy>
    && std::is_trivially_copyable_v<typename ProductPolicy::ProductParameters>
    && std::is_trivially_copyable_v<typename ProductPolicy::Calendar>
    && std::is_trivially_copyable_v<typename ProductPolicy::PreparedProduct>
    && std::is_trivially_copyable_v<typename ProductPolicy::Handler>
    && requires(
        const typename StatePolicy::Parameters& model,
        const typename ProductPolicy::ProductParameters& parameters,
        const typename ProductPolicy::PreparedProduct& prepared,
        const typename StatePolicy::State& terminal,
        const typename ProductPolicy::Handler& const_handler,
        ProductPreparationContext context
    ) {
        {
            ProductPolicy::kObservationCoordinate
        } -> std::convertible_to<ObservationCoordinate>;
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
