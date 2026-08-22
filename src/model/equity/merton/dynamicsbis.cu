// Experimental autonomous implementation of exact Merton paths.
#include "model/equity/merton/dynamicsbis.cuh"

#include <cmath>

namespace ai_factory::workbench::merton {

// ===================== Model mathematical primitives =====================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float jump_variance =
        parameters.jump_log_volatility * parameters.jump_log_volatility;
    const float jump_martingale = expf(
        parameters.jump_log_mean + 0.5f * jump_variance
    ) - 1.0f;
    const float diffusion_variance =
        parameters.volatility * parameters.volatility;
    return {
        logf(parameters.spot),
        parameters.risk_free_rate
            - parameters.dividend_yield
            - parameters.jump_intensity * jump_martingale
            - 0.5f * diffusion_variance,
        parameters.volatility,
        parameters.jump_intensity,
        parameters.jump_log_mean,
        parameters.jump_log_volatility,
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
    const PreparedModel& model,
    const PreparedTransition& transition,
    std::uint32_t jump_count,
    float diffusion_normal,
    float jump_normal,
    State& state
) {
    const float count = static_cast<float>(jump_count);
    const float jump_log_sum = count * model.jump_log_mean
        + model.jump_log_volatility * sqrtf(count) * jump_normal;
    state.log_spot += transition.drift
        + transition.diffusion_standard_deviation * diffusion_normal
        + jump_log_sum;
}

namespace {

__device__ __forceinline__ void simulate_exact_transition(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::NormalRandomContext& random,
    State& state
) {
    constexpr float kPoissonInversionThreshold = 10.0f;
    const std::uint32_t jump_count =
        transition.poisson_mean < kPoissonInversionThreshold
        ? philox::poisson_from_uniform(
            random.uniforms.next(),
            transition.poisson_mean,
            transition.zero_jump_probability
        )
        : philox::poisson_from_uniform_sequence(
            random.uniforms,
            transition.poisson_mean
        );
    const float diffusion_normal = philox::next_normal(
        random.uniforms,
        random.normals
    );
    const float jump_normal = jump_count == 0U
        ? 0.0f
        : philox::next_normal(random.uniforms, random.normals);
    one_step_transition(
        model,
        transition,
        jump_count,
        diffusion_normal,
        jump_normal,
        state
    );
}

}  // namespace

// ======================== Compile-time policy ============================

__device__ __forceinline__ DynamicsPolicy::PreparedDynamics
DynamicsPolicy::prepare_dynamics(
    const Parameters& parameters,
    float delta_t
) {
    const PreparedModel model = DynamicsPolicy::prepare_model(parameters);
    return {
        model,
        DynamicsPolicy::prepare_transition(model, delta_t),
    };
}

__device__ __forceinline__ DynamicsPolicy::PreparedModel
DynamicsPolicy::prepare_model(const Parameters& parameters) {
    return merton::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    return merton::prepare_transition(model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return merton::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& model) {
    return merton::initial_state(model);
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedDynamics& dynamics,
    RandomContext& random,
    State& state
) {
    DynamicsPolicy::simulate_one_step(
        dynamics.model,
        dynamics.transition,
        random,
        state
    );
}

__device__ __forceinline__ void DynamicsPolicy::advance(
    const PreparedDynamics& dynamics,
    std::uint32_t step_count,
    RandomContext& random,
    State& state
) {
    for (std::uint32_t step = 0U; step < step_count; ++step) {
        simulate_one_step(dynamics, random, state);
    }
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedModel& model,
    const PreparedTransition& transition,
    RandomContext& random,
    State& state
) {
    simulate_exact_transition(model, transition, random, state);
}

__device__ __forceinline__ float DynamicsPolicy::spot(
    const State& state
) {
    return expf(state.log_spot);
}

__device__ __forceinline__ float DynamicsPolicy::log_spot(
    const State& state
) {
    return state.log_spot;
}

__device__ __forceinline__ float DynamicsPolicy::risk_free_rate(
    const Parameters& parameters
) {
    return parameters.risk_free_rate;
}

}  // namespace ai_factory::workbench::merton
