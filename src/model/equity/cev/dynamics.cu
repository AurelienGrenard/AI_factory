#include "model/equity/cev/dynamics.cuh"

#include <cmath>

namespace ai_factory::workbench::cev {

__device__ __forceinline__ CevPreparedParameters prepare_model(
    const CevModelParameters& p, float maturity, std::size_t num_steps
) {
    const float dt = maturity / static_cast<float>(num_steps);
    return {
        p.spot,
        (p.risk_free_rate - p.dividend_yield) * dt,
        p.sigma * sqrtf(dt),
        0.5f * p.sigma * p.sigma * p.beta * dt,
        p.beta,
    };
}

__device__ __forceinline__ CevState initial_state(
    const CevPreparedParameters& model
) {
    return {model.initial_spot};
}

// Milstein preserves the local-volatility derivative term. Zero is absorbing,
// and a negative numerical proposal is projected to that attainable boundary.
__device__ __forceinline__ void one_step_transition(
    const CevPreparedParameters& model, float normal, CevState& state
) {
    const float spot = state.spot;
    if (!(spot > 0.0f)) {
        state.spot = 0.0f;
        return;
    }
    const float spot_beta = powf(spot, model.beta);
    // Reuse S^beta instead of evaluating a second power function:
    // S^(2 beta - 1) = (S^beta)^2 / S while S is strictly positive.
    const float spot_milstein_power = spot_beta * spot_beta / spot;
    const float proposal = spot
        + model.drift_dt * spot
        + model.diffusion_scale * spot_beta * normal
        + model.milstein_scale * spot_milstein_power * (normal * normal - 1.0f);
    state.spot = fmaxf(proposal, 0.0f);
}

namespace {
__device__ __forceinline__ void simulate_one_step(
    const CevPreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    CevState& state
) { one_step_transition(model, philox::next_normal(uniforms, normals), state); }

__device__ __forceinline__ void simulate_steps(
    const CevPreparedParameters& model,
    std::size_t num_steps,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normals,
    CevState& state
) {
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
    }
}
}  // namespace

__device__ __forceinline__ CevState simulate_terminal_state(
    const CevPreparedParameters& model, philox::PhiloxKey key,
    std::size_t path, std::size_t num_steps
) {
    CevState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(model, num_steps, uniforms, normals, state);
    return state;
}

__device__ __forceinline__ CevMeanPathResult simulate_mean_state(
    const CevPreparedParameters& model, philox::PhiloxKey key,
    std::size_t path, std::size_t num_steps
) {
    CevState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = state.spot;
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        sum += state.spot;
    }
    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {static_cast<float>(sum / observation_count)};
}

__device__ __forceinline__ CevGeometricMeanPathResult simulate_geometric_mean_state(
    const CevPreparedParameters& model, philox::PhiloxKey key,
    std::size_t path, std::size_t num_steps
) {
    CevState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    double sum = logf(state.spot);
    bool hit_zero = false;
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
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

__device__ __forceinline__ CevTwoTimePathResult simulate_at_two_times(
    const CevPreparedParameters& first, const CevPreparedParameters& second,
    philox::PhiloxKey key, std::size_t path,
    std::size_t first_num_steps, std::size_t second_num_steps
) {
    CevState state = initial_state(first);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(first, first_num_steps, uniforms, normals, state);
    const float first_spot = state.spot;
    simulate_steps(second, second_num_steps, uniforms, normals, state);
    return {first_spot, state.spot};
}

__device__ __forceinline__ CevMaximumPathResult simulate_maximum_state(
    const CevPreparedParameters& model, philox::PhiloxKey key,
    std::size_t path, std::size_t num_steps
) {
    CevState state = initial_state(model);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    float maximum = state.spot;
    for (std::size_t step = 0U; step < num_steps; ++step) {
        simulate_one_step(model, uniforms, normals, state);
        maximum = fmaxf(maximum, state.spot);
    }
    return {maximum};
}

__device__ __forceinline__ CevState simulate_on_regular_grid(
    const CevPreparedParameters& stub, const CevPreparedParameters& regular,
    philox::PhiloxKey key, std::size_t path,
    std::uint32_t initial_stub_steps, std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count, std::size_t path_count,
    float* observed_spots
) {
    CevState state = initial_state(stub);
    philox::UniformSequence uniforms(key, path);
    philox::NormalPairCache normals;
    simulate_steps(stub, initial_stub_steps, uniforms, normals, state);
    if (exercise_count == 1U) {
        return state;
    }
    std::size_t output = path;
    observed_spots[output] = state.spot;
    for (std::uint32_t exercise = 1U;
         exercise + 1U < exercise_count;
         ++exercise) {
        simulate_steps(regular, steps_per_exercise, uniforms, normals, state);
        output += path_count;
        observed_spots[output] = state.spot;
    }
    simulate_steps(regular, steps_per_exercise, uniforms, normals, state);
    return state;
}

}  // namespace ai_factory::workbench::cev
