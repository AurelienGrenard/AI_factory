#include "model/equity/variance_gamma/dynamicsbis.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::variance_gamma {

// ======================== Common equity dynamics =========================

__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters
) {
    const float sigma2 = parameters.sigma * parameters.sigma;
    const float martingale_argument = 1.0f
        - parameters.theta * parameters.nu
        - 0.5f * sigma2 * parameters.nu;
    const float martingale_correction =
        logf(martingale_argument) / parameters.nu;
    return {
        logf(parameters.spot),
        parameters.risk_free_rate - parameters.dividend_yield
            + martingale_correction,
        1.0f / parameters.nu,
        parameters.nu,
        parameters.theta,
        parameters.sigma,
    };
}

__device__ __forceinline__ PreparedTransition prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    return {
        model.drift_rate * delta_t,
        delta_t * model.inverse_nu,
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
    const PreparedModel& model,
    const PreparedTransition& transition,
    float gamma_increment,
    float brownian_normal,
    State& state
) {
    const float brownian_increment =
        model.sigma * sqrtf(gamma_increment) * brownian_normal;
    state.log_spot += fmaf(
        model.theta,
        gamma_increment,
        transition.drift + brownian_increment
    );
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
    const float gamma_increment = philox::marsaglia_tsang_gamma(
        uniforms,
        normals,
        transition.gamma_shape,
        model.nu
    );
    const float brownian_normal = philox::next_normal(uniforms, normals);
    one_step_transition(
        model, transition, gamma_increment, brownian_normal, state
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
    return variance_gamma::prepare_model(parameters);
}

__device__ __forceinline__ DynamicsPolicy::PreparedTransition
DynamicsPolicy::prepare_transition(
    const PreparedModel& model,
    float delta_t
) {
    return variance_gamma::prepare_transition(model, delta_t);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedDynamics& dynamics) {
    return variance_gamma::initial_state(dynamics.model);
}

__device__ __forceinline__ DynamicsPolicy::State
DynamicsPolicy::initial_state(const PreparedModel& model) {
    return variance_gamma::initial_state(model);
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
        DynamicsPolicy::simulate_one_step(dynamics, random, state);
    }
}

__device__ __forceinline__ void DynamicsPolicy::simulate_one_step(
    const PreparedModel& model,
    const PreparedTransition& transition,
    RandomContext& random,
    State& state
) {
    variance_gamma::simulate_one_step(
        model,
        transition,
        random.uniforms,
        random.normals,
        state
    );
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

}  // namespace ai_factory::workbench::variance_gamma

