// Reusable exact Variance-Gamma preparation and path simulation.
#include "model/equity/variance_gamma/dynamics.cuh"

#include "common/philox.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::variance_gamma {

// Prepare the exact Gamma clock increment over one requested interval.
__device__ __forceinline__ VarianceGammaPreparedParameters prepare_model(
    const VarianceGammaModelParameters& parameters,
    float time_interval
) {
    const float sigma2 = parameters.sigma * parameters.sigma;
    const float martingale_argument = 1.0f
        - parameters.theta * parameters.nu
        - 0.5f * sigma2 * parameters.nu;
    const float martingale_correction =
        logf(martingale_argument) / parameters.nu;
    const float drift_dt = (
        parameters.risk_free_rate
        - parameters.dividend_yield
        + martingale_correction
    ) * time_interval;

    return {
        logf(parameters.spot),
        time_interval / parameters.nu,
        parameters.nu,
        parameters.theta,
        parameters.sigma,
        drift_dt,
    };
}

// Prepare the exact law of one sub-step for a genuinely monitored grid.
__device__ __forceinline__ VarianceGammaPreparedParameters prepare_model(
    const VarianceGammaModelParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    return prepare_model(
        parameters, maturity / static_cast<float>(num_steps)
    );
}

// Construct the time-zero state stored in the prepared parameters.
__device__ __forceinline__ VarianceGammaState initial_state(
    const VarianceGammaPreparedParameters& model
) {
    return {model.initial_log_spot};
}

// Apply theta*G + sigma*sqrt(G)*Z and the risk-neutral drift exactly in law.
__device__ __forceinline__ void one_step_transition(
    const VarianceGammaPreparedParameters& model,
    float gamma_increment,
    float brownian_normal,
    VarianceGammaState& state
) {
    const float brownian_increment =
        model.sigma * sqrtf(gamma_increment) * brownian_normal;
    state.log_spot += fmaf(
        model.theta,
        gamma_increment,
        model.drift_dt + brownian_increment
    );
}

namespace {

// Draw one exact VG increment from the path-local scalar uniform sequence.
__device__ __forceinline__ void simulate_one_step(
    const VarianceGammaPreparedParameters& model,
    philox::UniformSequence& uniforms,
    philox::NormalPairCache& normal_cache,
    VarianceGammaState& state
) {
    const float gamma_increment = philox::marsaglia_tsang_gamma(
        uniforms,
        normal_cache,
        model.gamma_shape,
        model.gamma_scale
    );
    const float brownian_normal =
        philox::next_normal(uniforms, normal_cache);
    one_step_transition(
        model, gamma_increment, brownian_normal, state
    );
}

}  // namespace

// Generate all random variates for one path and return its terminal state.
__device__ __forceinline__ VarianceGammaState simulate_terminal_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path
) {
    VarianceGammaState state = initial_state(model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    simulate_one_step(model, uniforms, normal_cache, state);
    return state;
}

// Accumulate spots from time zero to maturity and return their arithmetic mean.
__device__ __forceinline__ VarianceGammaMeanPathResult simulate_mean_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    VarianceGammaState state = initial_state(model);
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
        static_cast<float>(
            spot_sum / (static_cast<double>(num_steps) + 1.0)
        ),
    };
}

// Average log-spots in FP64 and exponentiate only the completed mean.
__device__ __forceinline__ VarianceGammaGeometricMeanPathResult
simulate_geometric_mean_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    VarianceGammaState state = initial_state(model);
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
        expf(static_cast<float>(log_spot_sum / observation_count)),
    };
}

// Reuse one scalar uniform sequence across two exact increment preparations.
__device__ __forceinline__ VarianceGammaTwoTimePathResult
simulate_at_two_times(
    const VarianceGammaPreparedParameters& first_model,
    const VarianceGammaPreparedParameters& second_model,
    philox::PhiloxKey key,
    std::size_t path
) {
    VarianceGammaState state = initial_state(first_model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;

    simulate_one_step(first_model, uniforms, normal_cache, state);
    const float first_spot = expf(state.log_spot);

    simulate_one_step(second_model, uniforms, normal_cache, state);
    return {first_spot, expf(state.log_spot)};
}

// Track the maximum spot at time zero and after every simulated transition.
__device__ __forceinline__ VarianceGammaMaximumPathResult
simulate_maximum_state(
    const VarianceGammaPreparedParameters& model,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    VarianceGammaState state = initial_state(model);
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

    return {maximum_spot};
}

// Write pre-maturity spots in a date-major grid and return terminal state.
__device__ __forceinline__ VarianceGammaState simulate_on_regular_grid(
    const VarianceGammaPreparedParameters& initial_stub_model,
    const VarianceGammaPreparedParameters& regular_model,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_spots
) {
    VarianceGammaState state = initial_state(initial_stub_model);
    philox::UniformSequence uniforms(
        key, static_cast<std::uint64_t>(path)
    );
    philox::NormalPairCache normal_cache;
    simulate_one_step(initial_stub_model, uniforms, normal_cache, state);
    if (exercise_count == 1U) return state;
    std::size_t output_index = path;
    observed_spots[output_index] = expf(state.log_spot);

    for (std::uint32_t exercise = 1U;
         exercise + 1U < exercise_count;
         ++exercise) {
        simulate_one_step(regular_model, uniforms, normal_cache, state);
        output_index += path_count;
        observed_spots[output_index] = expf(state.log_spot);
    }

    simulate_one_step(regular_model, uniforms, normal_cache, state);
    return state;
}

}  // namespace ai_factory::workbench::variance_gamma
