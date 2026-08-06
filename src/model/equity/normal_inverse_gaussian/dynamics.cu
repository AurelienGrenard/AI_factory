// Reusable exact Normal-Inverse-Gaussian preparation and path simulation.
#include "model/equity/normal_inverse_gaussian/dynamics.cuh"

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::normal_inverse_gaussian {

// Prepare the exact inverse-Gaussian clock and martingale correction per step.
__device__ __forceinline__ NormalInverseGaussianPreparedParameters
prepare_model(
    const NormalInverseGaussianModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    const float dt = maturity / static_cast<float>(num_steps);
    const float alpha2 = parameters.alpha * parameters.alpha;
    const float beta2 = parameters.beta * parameters.beta;
    const float gamma = sqrtf(alpha2 - beta2);
    const float beta_plus_one = parameters.beta + 1.0f;
    const float exponential_moment_root = sqrtf(
        alpha2 - beta_plus_one * beta_plus_one
    );
    const float martingale_correction = parameters.delta
        * (exponential_moment_root - gamma);
    const float drift_dt = (
        parameters.risk_free_rate
        - parameters.dividend_yield
        + martingale_correction
    ) * dt;
    const float delta_dt = parameters.delta * dt;

    return {
        logf(parameters.spot),
        delta_dt / gamma,
        delta_dt * delta_dt,
        parameters.beta,
        drift_dt,
    };
}

// Construct the time-zero state stored in the prepared parameters.
__device__ __forceinline__ NormalInverseGaussianState initial_state(
    const NormalInverseGaussianPreparedParameters& model
) {
    return {model.initial_log_spot};
}

// Apply beta*G + sqrt(G)*Z and the risk-neutral drift exactly in law.
__device__ __forceinline__ void one_step_transition(
    const NormalInverseGaussianPreparedParameters& model,
    float inverse_gaussian_increment,
    float brownian_normal,
    NormalInverseGaussianState& state
) {
    const float brownian_increment =
        sqrtf(inverse_gaussian_increment) * brownian_normal;
    state.log_spot += fmaf(
        model.beta,
        inverse_gaussian_increment,
        model.drift_dt + brownian_increment
    );
}

namespace {

// Draw one exact NIG increment from the path-local scalar uniform sequence.
__device__ __forceinline__ void simulate_one_step(
    const NormalInverseGaussianPreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    NormalInverseGaussianState& state
) {
    const float inverse_gaussian_increment =
        philox::michael_schucany_haas_inverse_gaussian(
            uniforms,
            normal_cache,
            model.inverse_gaussian_mean,
            model.inverse_gaussian_shape
        );
    const float brownian_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        model, inverse_gaussian_increment, brownian_normal, state
    );
}

// Sum equal NIG increments analytically when only an interval boundary matters.
__device__ __forceinline__ NormalInverseGaussianPreparedParameters
aggregate_increment(
    const NormalInverseGaussianPreparedParameters& model,
    std::size_t step_count
) {
    const float count = static_cast<float>(step_count);
    NormalInverseGaussianPreparedParameters aggregate = model;
    aggregate.inverse_gaussian_mean *= count;
    aggregate.inverse_gaussian_shape *= count * count;
    aggregate.drift_dt *= count;
    return aggregate;
}

}  // namespace

// Generate all random variates for one path and return its terminal state.
__device__ __forceinline__ NormalInverseGaussianState simulate_terminal_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    NormalInverseGaussianState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    const NormalInverseGaussianPreparedParameters terminal_model =
        aggregate_increment(model, num_steps);
    simulate_one_step(terminal_model, uniforms, normal_cache, state);
    return state;
}

// Accumulate spots from time zero to maturity and return their arithmetic mean.
__device__ __forceinline__ NormalInverseGaussianMeanPathResult
simulate_mean_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    NormalInverseGaussianState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    double spot_sum = static_cast<double>(expf(state.log_spot));

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        spot_sum += static_cast<double>(expf(state.log_spot));
    }

    return {
        state,
        static_cast<float>(
            spot_sum / (static_cast<double>(num_steps) + 1.0)
        ),
    };
}

// Average log-spots in FP64 and exponentiate only the completed mean.
__device__ __forceinline__ NormalInverseGaussianGeometricMeanPathResult
simulate_geometric_mean_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    NormalInverseGaussianState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    double log_spot_sum = static_cast<double>(state.log_spot);

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        log_spot_sum += static_cast<double>(state.log_spot);
    }

    const double observation_count = static_cast<double>(num_steps) + 1.0;
    return {
        state,
        expf(static_cast<float>(log_spot_sum / observation_count)),
    };
}

// Reuse one scalar uniform sequence across two exact increment preparations.
__device__ __forceinline__ NormalInverseGaussianTwoTimePathResult
simulate_at_two_times(
    const NormalInverseGaussianPreparedParameters& first_model,
    const NormalInverseGaussianPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t first_num_steps,
    std::size_t second_num_steps
) {
    NormalInverseGaussianState state = initial_state(first_model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    const NormalInverseGaussianPreparedParameters first_interval =
        aggregate_increment(first_model, first_num_steps);
    simulate_one_step(first_interval, uniforms, normal_cache, state);
    const NormalInverseGaussianState first_state = state;

    const NormalInverseGaussianPreparedParameters second_interval =
        aggregate_increment(second_model, second_num_steps);
    simulate_one_step(second_interval, uniforms, normal_cache, state);
    return {first_state, state};
}

// Track the maximum spot at time zero and after every simulated transition.
__device__ __forceinline__ NormalInverseGaussianMaximumPathResult
simulate_maximum_state(
    const NormalInverseGaussianPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    NormalInverseGaussianState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    float maximum_spot = expf(state.log_spot);

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        simulate_one_step(model, uniforms, normal_cache, state);
        maximum_spot = fmaxf(maximum_spot, expf(state.log_spot));
    }

    return {state, maximum_spot};
}

// Write pre-maturity spots in a date-major grid and return terminal state.
__device__ __forceinline__ NormalInverseGaussianState
simulate_on_regular_grid(
    const NormalInverseGaussianPreparedParameters& initial_stub_model,
    const NormalInverseGaussianPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
) {
    NormalInverseGaussianState state = initial_state(initial_stub_model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    const NormalInverseGaussianPreparedParameters initial_interval =
        aggregate_increment(initial_stub_model, initial_stub_steps);
    simulate_one_step(initial_interval, uniforms, normal_cache, state);
    if (exercise_count == 1U) return state;
    std::size_t output_index = path;
    observed_spots[output_index] = expf(state.log_spot);

    const NormalInverseGaussianPreparedParameters regular_interval =
        aggregate_increment(regular_model, steps_per_exercise);

    for (std::uint32_t exercise = 1U;
         exercise + 1U < exercise_count;
         ++exercise) {
        simulate_one_step(
            regular_interval, uniforms, normal_cache, state
        );
        output_index += path_count;
        observed_spots[output_index] = expf(state.log_spot);
    }

    simulate_one_step(regular_interval, uniforms, normal_cache, state);
    return state;
}

}  // namespace ai_factory::workbench::normal_inverse_gaussian
