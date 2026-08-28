// Device transition definitions for the quadratic rough Heston N-factor dynamics policy.
#pragma once

#include "model/equity/rough/quadratic_rough_heston/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::model::equity::quadratic_rough_heston {

namespace detail {

template<std::size_t FactorCount>
__device__ __forceinline__ float feedback(
    const PreparedDynamics<FactorCount>& dynamics,
    const State<FactorCount>& state
) {
    float value = dynamics.initial_feedback;
    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        value = fmaf(
            dynamics.kernel.weights[factor],
            state.feedback_factors[factor],
            value
        );
    }
    return value;
}

template<std::size_t FactorCount>
__device__ __forceinline__ float variance(
    const PreparedDynamics<FactorCount>& dynamics,
    const State<FactorCount>& state
) {
    const float centered = feedback(dynamics, state)
        - dynamics.quadratic_shift;
    return fmaf(
        dynamics.quadratic_scale,
        centered * centered,
        dynamics.variance_floor
    );
}

}  // namespace detail

template<std::size_t FactorCount>
__device__ __forceinline__ auto DynamicsPolicy<FactorCount>::initial_state(
    const PreparedDynamics& dynamics
) -> State {
    State state{};
    state.log_spot = dynamics.initial_log_spot;
    return state;
}

template<std::size_t FactorCount>
__device__ __forceinline__ void DynamicsPolicy<FactorCount>::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    const float normal = philox::next_normal(random.uniforms, random.normals);
    const float feedback = detail::feedback(dynamics, state);
    const float variance = detail::variance(dynamics, state);
    const float sqrt_variance = sqrtf(fmaxf(variance, 0.0f));
    state.log_spot += dynamics.drift_dt - 0.5f * variance * dynamics.dt
        + sqrt_variance * dynamics.sqrt_dt * normal;

    // A raw explicit cell can multiply the quadratic-feedback state by a
    // large Gaussian factor when H is small.  Balance the complete Volterra
    // cell contribution smoothly: x / hypot(1, x) is bounded, has no branch
    // or arbitrary clipping threshold, and differs from x only at cubic
    // order as the cell size tends to zero.
    const float raw_force = fmaf(
        dynamics.feedback_rate * dynamics.feedback_volatility
            * dynamics.inverse_sqrt_dt,
        sqrt_variance * normal,
        -dynamics.feedback_rate * feedback
    );
    const float raw_cell_increment =
        dynamics.feedback_cell_loading * raw_force;
    const float balanced_force = raw_force
        / hypotf(1.0f, raw_cell_increment);

    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        state.feedback_factors[factor] = fmaf(
            dynamics.factor_decay[factor],
            state.feedback_factors[factor],
            dynamics.factor_drift_integral[factor] * balanced_force
        );
    }
}

template<std::size_t FactorCount>
__device__ __forceinline__ void DynamicsPolicy<FactorCount>::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        simulate_one_step(dynamics, random, state);
    }
}

template<std::size_t FactorCount>
__device__ __forceinline__ float DynamicsPolicy<FactorCount>::spot(
    const State& state
) {
    return expf(state.log_spot);
}

template<std::size_t FactorCount>
__device__ __forceinline__ float DynamicsPolicy<FactorCount>::log_spot(
    const State& state
) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::model::equity::quadratic_rough_heston
