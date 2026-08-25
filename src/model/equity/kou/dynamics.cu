#include "model/equity/kou/dynamics.cuh"

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench::kou {

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float jump_martingale = parameters.up_probability
            / (parameters.positive_jump_rate - 1.0f)
        - (1.0f - parameters.up_probability)
            / (parameters.negative_jump_rate + 1.0f);
    const float diffusion_variance =
        parameters.volatility * parameters.volatility;
    const float carry = parameters.risk_free_rate
        - parameters.dividend_yield - 0.5f * diffusion_variance;
    return {
        logf(parameters.spot),
        fmaf(-parameters.jump_intensity, jump_martingale, carry),
        parameters.volatility,
        parameters.jump_intensity,
        parameters.up_probability,
        1.0f / parameters.positive_jump_rate,
        1.0f / parameters.negative_jump_rate,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    const float poisson_mean = model.jump_intensity * delta_t;
    return {
        model.drift_rate * delta_t,
        model.volatility * sqrtf(delta_t),
        poisson_mean,
        expf(-poisson_mean),
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& model
) {
    return {model.initial_log_spot};
}

__device__ __forceinline__ void one_step_transition(
    const PreparedTransition& transition,
    float diffusion_normal,
    float jump_log_sum,
    State& state
) {
    state.log_spot += transition.drift
        + transition.diffusion_standard_deviation * diffusion_normal
        + jump_log_sum;
}

// ==================== Model-specific implementation =======================

namespace {

__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) {
    constexpr float kPoissonInversionThreshold = 10.0f;
    const std::uint32_t jump_count =
        transition.poisson_mean < kPoissonInversionThreshold
        ? philox::poisson_from_uniform(
            uniforms.next(),
            transition.poisson_mean,
            transition.zero_jump_probability
        )
        : philox::poisson_from_uniform_sequence(
            uniforms,
            transition.poisson_mean
        );
    float jump_log_sum = 0.0f;
    for (std::uint32_t jump_index = 0U;
         jump_index < jump_count;
         ++jump_index) {
        const bool upward = uniforms.next() < model.up_probability;
        const float magnitude = -logf(uniforms.next());
        jump_log_sum += upward
            ? magnitude * model.inverse_positive_jump_rate
            : -magnitude * model.inverse_negative_jump_rate;
    }
    one_step_transition(
        transition,
        philox::next_normal(uniforms, normals),
        jump_log_sum,
        state
    );
}

}  // namespace

// ======================== Common equity dynamics =========================

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    const PreparedModel model = DynamicsPolicy::prepare_model(parameters);
    return {model, DynamicsPolicy::prepare_transition(model, delta_t)};
}

__device__ __forceinline__ DynamicsPolicy::PreparedModel
DynamicsPolicy::prepare_model(const Parameters& parameters) {
    return kou::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    return kou::prepare_transition(model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return kou::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& model) {
    return kou::initial_state(model);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    DynamicsPolicy::simulate_one_step(
        dynamics.model, dynamics.transition, random, state
    );
}

__device__ __forceinline__ void DynamicsPolicy::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        DynamicsPolicy::simulate_one_step(dynamics, random, state);
    }
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedModel& model,
    const PreparedTransition& transition,
    RandomContext& random,
    State& state
) {
    kou::simulate_one_step(
        model, transition, random.uniforms, random.normals, state
    );
}

__device__ __forceinline__ float DynamicsPolicy::spot(const State& state) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float DynamicsPolicy::log_spot(
    const State& state
) {
    return state.log_spot;
}

}  // namespace ai_factory::workbench::kou
