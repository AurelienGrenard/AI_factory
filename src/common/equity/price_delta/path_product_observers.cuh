// Three payoff observers freeze independently while their shared schedule continues.
#pragma once
#include "common/equity/path_product_policy.cuh"
#include "common/equity/price_delta/spot_bump.cuh"

namespace ai_factory::workbench::equity::price_delta {

template<typename ProductPathPolicy, typename PathPolicy>
struct PairedProductObservers {
    using Dynamics = typename PathPolicy::Dynamics;
    struct ScenarioObserver {
        typename ProductPathPolicy::Handler handler;
        SpotObservation terminal{};
        bool active = true;

        __device__ __forceinline__ static float coordinate(SpotObservation state) {
            if constexpr (ProductPathPolicy::kObservationCoordinate
                          == ObservationCoordinate::spot) return state.spot;
            else return state.log_spot;
        }
        template<bool Initial>
        __device__ __forceinline__ void observe(
            std::uint32_t observation, SpotObservation state
        ) {
            if (!active) return;
            terminal = state;
            if constexpr (Initial) active = handler.on_initial_value(coordinate(state));
            else active = handler.on_observation(observation, coordinate(state));
        }
    };

    struct Observer {
        const typename PathPolicy::Prepared& path;
        ScenarioObserver (&scenarios)[3];

        template<bool Initial>
        __device__ __forceinline__ bool observe(
            std::uint32_t observation, const typename Dynamics::State& state
        ) {
            scenarios[0].template observe<Initial>(observation,
                PathPolicy::template observe<0U>(path, state));
            scenarios[1].template observe<Initial>(observation,
                PathPolicy::template observe<1U>(path, state));
            scenarios[2].template observe<Initial>(observation,
                PathPolicy::template observe<2U>(path, state));
            return scenarios[0].active || scenarios[1].active || scenarios[2].active;
        }
        __device__ __forceinline__ bool on_initial_state(const typename Dynamics::State& state) {
            return observe<true>(0U, state);
        }
        __device__ __forceinline__ bool on_observation(
            std::uint32_t observation, const typename Dynamics::State& state
        ) {
            return observe<false>(observation, state);
        }
    };

};

}  // namespace ai_factory::workbench::equity::price_delta
