#include "model/equity/cev/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::cev {

// ======================== Common equity dynamics =========================

// Prepare the coefficients defining one transition of duration delta_t
// under the supplied model parameters.
__device__ __forceinline__ PreparedModel prepare_model(
    const ModelParameters& parameters, float delta_t
) {
    return {
        parameters.spot,
        (parameters.risk_free_rate - parameters.dividend_yield) * delta_t,
        parameters.sigma * sqrtf(delta_t),
        0.5f * parameters.sigma * parameters.sigma * parameters.beta
            * delta_t,
        parameters.beta,
    };
}

__device__ __forceinline__ State initial_state(
    const PreparedModel& prepared_model
) {
    return {prepared_model.initial_spot};
}

// Milstein preserves the local-volatility derivative term. Zero is absorbing,
// and a negative numerical proposal is projected to that attainable boundary.
__device__ __forceinline__ void one_step_transition(
    const PreparedModel& prepared_model, float normal, State& state
) {
    const float spot = state.spot;
    if (!(spot > 0.0f)) {
        state.spot = 0.0f;
        return;
    }
    const float spot_beta = powf(spot, prepared_model.beta);
    // Reuse S^beta instead of evaluating a second power function:
    // S^(2 beta - 1) = (S^beta)^2 / S while S is strictly positive.
    const float spot_milstein_power = spot_beta * spot_beta / spot;
    const float proposal = spot
        + prepared_model.drift_dt * spot
        + prepared_model.diffusion_scale * spot_beta * normal
        + prepared_model.milstein_scale * spot_milstein_power * (normal * normal - 1.0f);
    state.spot = fmaxf(proposal, 0.0f);
}

// ==================== Model-specific implementation =======================

namespace {
__device__ __forceinline__ void simulate_one_step(
    const PreparedModel& prepared_model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) { one_step_transition(prepared_model, philox::next_normal(uniforms, normals), state); }

__device__ __forceinline__ void simulate_steps(
    const PreparedModel& prepared_model,
    std::uint32_t num_steps,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    State& state
) {
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
    }
}
}  // namespace

// ======================== Common equity dynamics =========================

__device__ __forceinline__ State simulate_terminal_state(
    const PreparedModel& prepared_model, philox::PhiloxKey key,
    std::size_t path, std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(prepared_model, num_steps, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ MeanPathResult simulate_mean_state(
    const PreparedModel& prepared_model, philox::PhiloxKey key,
    std::size_t path, std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = state.spot;
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
        sum += state.spot;
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {static_cast<float>(sum / observation_count)};
}

__device__ __forceinline__ GeometricMeanPathResult simulate_geometric_mean_state(
    const PreparedModel& prepared_model, philox::PhiloxKey key,
    std::size_t path, std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = logf(state.spot);
    bool hit_zero = false;
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
        hit_zero = hit_zero || !(state.spot > 0.0f);
        if (!hit_zero) {
            sum += logf(state.spot);
        }
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    const float mean = hit_zero
        ? 0.0f
        : expf(static_cast<float>(sum / observation_count));
    return {mean};
}

__device__ __forceinline__ MaximumPathResult simulate_maximum_state(
    const PreparedModel& prepared_model, philox::PhiloxKey key,
    std::size_t path, std::uint32_t num_steps
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = state.spot;
    for (std::uint32_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(prepared_model, uniforms, normals, state);
        maximum = fmaxf(maximum, state.spot);
    }
    return {maximum};
}

__device__ __forceinline__ State simulate_on_regular_grid(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key, std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_observation,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(
        prepared_model, initial_stub_steps, uniforms, normals, state
    );
    if (observation_count == 1U) {
        return state;
    }
    std::size_t output = 0U;
    observed_spots[output] = state.spot;
    for (std::uint32_t observation = 1U;
         observation + 1U < observation_count;
         ++observation) {
        simulate_steps(
            prepared_model,
            steps_per_observation,
            uniforms,
            normals,
            state
        );
        output += observation_stride;
        observed_spots[output] = state.spot;
    }
    simulate_steps(
        prepared_model,
        steps_per_observation,
        uniforms,
        normals,
        state
    );
    return state;
}

__device__ __forceinline__ State simulate_on_calendar(
    const PreparedModel& prepared_model,
    philox::PhiloxKey key,
    std::size_t path,
    const std::uint32_t* __restrict__ steps_between_observations,
    std::uint32_t observation_count,
    std::size_t observation_stride,
    float* __restrict__ observed_spots
) {
    State state = initial_state(prepared_model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    std::size_t output = 0U;
    for (std::uint32_t observation = 0U;
         observation < observation_count;
         ++observation) {
        simulate_steps(
            prepared_model,
            steps_between_observations[observation],
            uniforms,
            normals,
            state
        );
        if (observation + 1U < observation_count) {
            observed_spots[output] = state.spot;
            output += observation_stride;
        }
    }
    return state;
}

}  // namespace ai_factory::workbench::cev
