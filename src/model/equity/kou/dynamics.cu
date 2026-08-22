#include "model/equity/kou/dynamics.cuh"

#include <cmath>
#include <cstddef>
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

__device__ __forceinline__ void prepare_calendar(
    const PreparedModel& model,
    const std::uint32_t* __restrict__ interval_steps,
    std::uint32_t interval_count,
    float delta_t,
    PreparedTransition* __restrict__ transitions
) {
    for (std::uint32_t interval = 0U;
         interval < interval_count;
         ++interval) {
        transitions[interval] = prepare_transition(
            model,
            static_cast<float>(interval_steps[interval]) * delta_t
        );
    }
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

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path
) {
    State state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(model, transition, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ MeanPathResult simulate_mean_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
) {
    State state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = expf(state.log_spot);
    for (std::uint32_t interval = 0U;
         interval < interval_count;
         ++interval) {
        simulate_one_step(model, transition, uniforms, normals, state);
        sum += expf(state.log_spot);
    }
    return {
        static_cast<float>(
            sum / (static_cast<double>(interval_count) + 1.0)
        ),
    };
}

__device__ __forceinline__ GeometricMeanPathResult
simulate_geometric_mean_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
) {
    State state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double log_sum = state.log_spot;
    for (std::uint32_t interval = 0U;
         interval < interval_count;
         ++interval) {
        simulate_one_step(model, transition, uniforms, normals, state);
        log_sum += state.log_spot;
    }
    return {
        expf(static_cast<float>(
            log_sum / (static_cast<double>(interval_count) + 1.0)
        )),
    };
}

__device__ __forceinline__ MaximumPathResult simulate_maximum_state(
    const PreparedModel& model,
    const PreparedTransition& transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t interval_count
) {
    State state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = expf(state.log_spot);
    for (std::uint32_t interval = 0U;
         interval < interval_count;
         ++interval) {
        simulate_one_step(model, transition, uniforms, normals, state);
        maximum = fmaxf(maximum, expf(state.log_spot));
    }
    return {maximum};
}

__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& model,
    const PreparedTransition* __restrict__ transitions,
    std::uint32_t observation_count,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
) {
    State state = initial_state(model);
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    for (std::uint32_t observation = 0U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, transitions[observation], uniforms, normals, state
        );
        observed_spots[
            static_cast<std::size_t>(observation) * observation_stride
        ] = expf(state.log_spot);
    }
    simulate_one_step(
        model,
        transitions[observation_count - 1U],
        uniforms,
        normals,
        state
    );
    return state;
}

__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& model,
    const PreparedTransition& initial_stub_transition,
    const PreparedTransition& regular_transition,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
) {
    State state = initial_state(model);
    if (observation_count == 0U) return state;
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_one_step(
        model, initial_stub_transition, uniforms, normals, state
    );
    if (observation_count == 1U) return state;
    observed_spots[0U] = expf(state.log_spot);
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_one_step(
            model, regular_transition, uniforms, normals, state
        );
        observed_spots[
            static_cast<std::size_t>(observation) * observation_stride
        ] = expf(state.log_spot);
    }
    simulate_one_step(
        model, regular_transition, uniforms, normals, state
    );
    return state;
}

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

__device__ __forceinline__ float DynamicsPolicy::risk_free_rate(
    const Parameters& parameters
) {
    return parameters.risk_free_rate;
}

}  // namespace ai_factory::workbench::kou
