// Share Gaussian-Volterra innovations; couple spot-dependent model states explicitly.
#pragma once
#include "common/equity/price_delta/multiplicative_spot_path.cuh"

namespace ai_factory::workbench::equity::price_delta {

template<typename ModelPath>
struct MultiplicativeVolterraSpotPath : MultiplicativeSpotPath<ModelPath> {
    using Dynamics = ModelPath;
    __device__ __forceinline__ static typename Dynamics::PreparedModel prepare_dynamics(
        const typename Dynamics::Parameters&, const typename Dynamics::PreparedModel& central,
        float, SpotBump
    ) { return central; }
};

// Used for rough SABR: xi_0 stays fixed, dimensional alpha is prepared at each S0.
template<typename ModelPath>
struct CoupledVolterraSpotPaths {
    using ModelParameters = typename ModelPath::Parameters;
    struct Prepared {};
    struct Dynamics {
        struct PreparedModel {
            typename ModelPath::PreparedModel central, lower, upper;
        };
        struct State { typename ModelPath::State central, lower, upper; };
        __device__ __forceinline__ static State initial_state(const PreparedModel& model) {
            return {ModelPath::initial_state(model.central), ModelPath::initial_state(model.lower),
                    ModelPath::initial_state(model.upper)};
        }
        __device__ __forceinline__ static void advance(const PreparedModel& model,
            float value, float variance, float rough_normal, float spot_normal, State& state) {
            ModelPath::advance(model.central, value, variance, rough_normal, spot_normal, state.central);
            ModelPath::advance(model.lower, value, variance, rough_normal, spot_normal, state.lower);
            ModelPath::advance(model.upper, value, variance, rough_normal, spot_normal, state.upper);
        }
    };
    __device__ __forceinline__ static Prepared prepare(SpotBump) { return {}; }
    __device__ __forceinline__ static typename Dynamics::PreparedModel prepare_dynamics(
        const ModelParameters& model, const typename ModelPath::PreparedModel& central,
        float dt, SpotBump bump) {
        auto lower = model, upper = model;
        lower.spot = bump.lower;
        upper.spot = bump.upper;
        return {central, ModelPath::prepare_model(lower, dt), ModelPath::prepare_model(upper, dt)};
    }
    template<unsigned Scenario>
    __device__ __forceinline__ static SpotObservation observe(const Prepared&, const typename Dynamics::State& state) {
        static_assert(Scenario < 3U);
        const auto& selected = [&]() -> const typename ModelPath::State& {
            if constexpr (Scenario == 0U) return state.central;
            else if constexpr (Scenario == 1U) return state.lower;
            else return state.upper;
        }();
        return {ModelPath::spot(selected), ModelPath::log_spot(selected)};
    }
};

}  // namespace ai_factory::workbench::equity::price_delta
