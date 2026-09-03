// Included device definitions of the weak second-order rough-Heston N-factor transition.
#pragma once

#include "model/equity/rough/rough_heston/dynamics.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::equity::rough_heston {
namespace detail {

constexpr float kSqrtThree = 1.7320508075688772935f;
constexpr float kWeakB = (6.0f + kSqrtThree) / 4.0f;
constexpr float kWeakA = kWeakB - 0.75f;

template<std::size_t FactorCount>
__device__ __forceinline__ float aggregate_variance(
    const PreparedDynamics<FactorCount>& dynamics,
    const State<FactorCount>& state
) {
    float variance = 0.0f;
    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        variance = fmaf(
            dynamics.kernel.weights[factor],
            state.variance_factors[factor],
            variance
        );
    }
    return fmaxf(variance, 0.0f);
}

template<std::size_t FactorCount>
__device__ __forceinline__ void ode_half_step(
    const PreparedDynamics<FactorCount>& dynamics,
    State<FactorCount>& state
) {
    float next[FactorCount];
    #pragma unroll
    for (std::size_t row = 0U; row < FactorCount; ++row) {
        float value = dynamics.ode_half_shift[row];
        #pragma unroll
        for (std::size_t column = 0U; column < FactorCount; ++column) {
            value = fmaf(
                dynamics.ode_half_step[row * FactorCount + column],
                state.variance_factors[column],
                value
            );
        }
        next[row] = value;
    }
    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor)
        state.variance_factors[factor] = next[factor];
}

template<std::size_t FactorCount>
__device__ __forceinline__ void weak_stochastic_variance_step(
    const PreparedDynamics<FactorCount>& dynamics,
    float uniform,
    State<FactorCount>& state
) {
    const float variance = aggregate_variance(dynamics, state);
    if (!(variance > 1.0e-12f)) return;
    const float z = dynamics.weak_variance_scale;
    const float bz = kWeakB * z;
    const float root = sqrtf(fmaf(3.0f * z, variance, bz * bz));
    const float first_denominator =
        (variance + bz - root) * root
        * (root - (kWeakB - kWeakA) * z);
    const float first_numerator = 0.5f * z * variance * (
        (kWeakA * kWeakB - kWeakA - kWeakB + 1.5f) * z
        + 0.25f * (kSqrtThree - 1.0f) * root
        + variance
    );
    const float second_denominator = 1.5f * variance
        + kWeakA * (kWeakB - 0.5f * kWeakA) * z;
    const float probability_first = fminf(
        fmaxf(first_numerator / first_denominator, 0.0f), 1.0f
    );
    const float probability_second = fminf(
        fmaxf(variance / second_denominator, 0.0f),
        1.0f - probability_first
    );

    float increment = kWeakA * z;
    if (uniform < probability_first) {
        increment = bz - root;
    } else if (uniform >= probability_first + probability_second) {
        increment = bz + root;
    }
    const float factor_increment = increment / dynamics.weight_sum;
    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor)
        state.variance_factors[factor] += factor_increment;
}

template<std::size_t FactorCount>
__device__ __forceinline__ void orthogonal_stock_half_step(
    const PreparedDynamics<FactorCount>& dynamics,
    float normal,
    State<FactorCount>& state
) {
    const float variance = aggregate_variance(dynamics, state);
    state.log_spot = fmaf(
        sqrtf(variance * dynamics.orthogonal_variance_scale),
        normal,
        state.log_spot - dynamics.orthogonal_drift_scale * variance
    );
}

template<std::size_t FactorCount>
__device__ __forceinline__ void correlated_stock_step(
    const PreparedDynamics<FactorCount>& dynamics,
    const float* previous_factors,
    State<FactorCount>& state
) {
    float weighted_sum = 0.0f;
    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        weighted_sum = fmaf(
            dynamics.kernel.weights[factor],
            previous_factors[factor] + state.variance_factors[factor],
            weighted_sum
        );
    }
    float correlated = dynamics.correlated_constant;
    correlated = fmaf(
        dynamics.half_dt_first_node,
        previous_factors[0] + state.variance_factors[0],
        correlated
    );
    correlated = fmaf(
        dynamics.correlated_weight_scale, weighted_sum, correlated
    );
    correlated += state.variance_factors[0] - previous_factors[0];
    state.log_spot += dynamics.spot_drift_dt
        + dynamics.rho_over_volatility * correlated;
}

}  // namespace detail

template<std::size_t FactorCount>
__device__ __forceinline__ typename DynamicsPolicy<FactorCount>::State
DynamicsPolicy<FactorCount>::initial_state(
    const PreparedDynamics& dynamics
) {
    State state{};
    state.log_spot = dynamics.initial_log_spot;
    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor)
        state.variance_factors[factor] = dynamics.initial_factors[factor];
    return state;
}

template<std::size_t FactorCount>
__device__ __forceinline__ void DynamicsPolicy<FactorCount>::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    // One normal, one independent uniform, then the cached second normal.
    const float first_normal = philox::next_normal(
        random.uniforms, random.normals
    );
    const float variance_uniform = random.uniforms.next();
    const float second_normal = philox::next_normal(
        random.uniforms, random.normals
    );

    detail::orthogonal_stock_half_step(dynamics, first_normal, state);
    float previous_factors[FactorCount];
    #pragma unroll
    for (std::size_t factor = 0U; factor < FactorCount; ++factor)
        previous_factors[factor] = state.variance_factors[factor];
    detail::ode_half_step(dynamics, state);
    detail::weak_stochastic_variance_step(
        dynamics, variance_uniform, state
    );
    detail::ode_half_step(dynamics, state);
    detail::correlated_stock_step(dynamics, previous_factors, state);
    detail::orthogonal_stock_half_step(dynamics, second_normal, state);
}

template<std::size_t FactorCount>
__device__ __forceinline__ void DynamicsPolicy<FactorCount>::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step)
        simulate_one_step(dynamics, random, state);
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

}  // namespace ai_factory::workbench::model::equity::rough_heston
