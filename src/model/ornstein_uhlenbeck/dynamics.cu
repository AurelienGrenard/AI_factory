// Exact Ornstein-Uhlenbeck factor and factor-integral simulation.
#include "model/ornstein_uhlenbeck/dynamics.cuh"

// Include OU analytics so exact transition coefficients can be inlined.
#include "model/ornstein_uhlenbeck/analytics.cu"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::model::ornstein_uhlenbeck {
namespace {

constexpr std::uint64_t kUniformsPerStep = 2ULL;

}  // namespace

// Prepare one exact joint Gaussian transition for x and its time integral.
__device__ __forceinline__ OrnsteinUhlenbeckExactParameters prepare_model(
    const OrnsteinUhlenbeckDynamicsParameters& parameters,
    float maturity,
    std::size_t num_steps
) {
    const float a = parameters.mean_reversion;
    const float sigma2 = parameters.volatility * parameters.volatility;
    const float dt = maturity / static_cast<float>(num_steps);
    const float decay = expf(-a * dt);
    const float one_minus_decay = -expm1f(-a * dt);
    const float factor_variance =
        sigma2 * (-expm1f(-2.0f * a * dt)) / (2.0f * a);
    const float factor_standard_deviation = sqrtf(factor_variance);
    const float covariance =
        sigma2 * one_minus_decay * one_minus_decay / (2.0f * a * a);
    const float integral_factor_normal_loading =
        covariance / factor_standard_deviation;
    const float independent_variance = fmaxf(
        integral_variance(parameters, dt)
            - integral_factor_normal_loading
                * integral_factor_normal_loading,
        0.0f
    );

    return {
        decay,
        factor_standard_deviation,
        one_minus_decay / a,
        integral_factor_normal_loading,
        sqrtf(independent_variance),
    };
}

// Start from x(0) = 0 and a zero accumulated factor integral.
__device__ __forceinline__ OrnsteinUhlenbeckState initial_state(
    float initial_factor
) {
    return {initial_factor, 0.0f};
}

// Simulate x and integral(x ds) jointly without a curve evaluation.
__device__ __forceinline__ void one_step_transition(
    const OrnsteinUhlenbeckExactParameters& model,
    float factor_normal,
    float integral_normal,
    OrnsteinUhlenbeckState& state
) {
    const float previous_factor = state.factor;
    const float factor_noise =
        model.factor_standard_deviation * factor_normal;
    const float integral_noise =
        model.integral_factor_normal_loading * factor_normal
        + model.integral_independent_standard_deviation * integral_normal;

    state.factor = model.decay * previous_factor + factor_noise;
    state.integrated_factor +=
        model.integral_factor_loading * previous_factor
        + integral_noise;
}

// Generate all random variates for one path and return its terminal state.
__device__ __forceinline__ OrnsteinUhlenbeckState simulate_terminal_state(
    const OrnsteinUhlenbeckExactParameters& model,
    float initial_factor,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    OrnsteinUhlenbeckState state = initial_state(initial_factor);
    const std::uint64_t uniform_count =
        kUniformsPerStep * static_cast<std::uint64_t>(num_steps);
    const std::uint64_t groups_per_path = (uniform_count + 3ULL) >> 2U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::UniformSequence uniforms(key, first_group);

    for (std::size_t step_index = 0U;
         step_index < num_steps;
         ++step_index) {
        const float angle_uniform = uniforms.next();
        const float radius_uniform = uniforms.next();
        const philox::NormalPair normals =
            philox::box_muller(angle_uniform, radius_uniform);
        one_step_transition(model, normals.first, normals.second, state);
    }
    return state;
}

// Write pre-maturity states in a date-major grid and return terminal state.
__device__ __forceinline__ OrnsteinUhlenbeckState simulate_on_regular_grid(
    const OrnsteinUhlenbeckExactParameters& initial_stub_model,
    const OrnsteinUhlenbeckExactParameters& regular_model,
    float initial_factor,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_factors,
    float* __restrict__ observed_integrated_factors
) {
    OrnsteinUhlenbeckState state = initial_state(initial_factor);
    const std::uint64_t total_steps =
        static_cast<std::uint64_t>(initial_stub_steps)
        + static_cast<std::uint64_t>(exercise_count - 1U)
            * static_cast<std::uint64_t>(steps_per_exercise);
    const std::uint64_t uniform_count = kUniformsPerStep * total_steps;
    const std::uint64_t groups_per_path = (uniform_count + 3ULL) >> 2U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::UniformSequence uniforms(key, first_group);

    // Reach the first exercise date through its possibly shorter stub.
    for (std::uint32_t step_index = 0U;
         step_index < initial_stub_steps;
         ++step_index) {
        const float angle_uniform = uniforms.next();
        const float radius_uniform = uniforms.next();
        const philox::NormalPair normals =
            philox::box_muller(angle_uniform, radius_uniform);
        one_step_transition(
            initial_stub_model,
            normals.first,
            normals.second,
            state
        );
    }
    if (exercise_count > 1U) {
        observed_factors[path] = state.factor;
        observed_integrated_factors[path] = state.integrated_factor;
    }

    // Every remaining exercise interval uses the same numerical grid.
    for (std::uint32_t exercise = 1U;
         exercise < exercise_count;
         ++exercise) {
        for (std::uint32_t step_index = 0U;
             step_index < steps_per_exercise;
             ++step_index) {
            const float angle_uniform = uniforms.next();
            const float radius_uniform = uniforms.next();
            const philox::NormalPair normals =
                philox::box_muller(angle_uniform, radius_uniform);
            one_step_transition(
                regular_model,
                normals.first,
                normals.second,
                state
            );
        }

        if (exercise + 1U < exercise_count) {
            const std::size_t output_index =
                static_cast<std::size_t>(exercise) * path_count + path;
            observed_factors[output_index] = state.factor;
            observed_integrated_factors[output_index] =
                state.integrated_factor;
        }
    }
    return state;
}

}  // namespace ai_factory::workbench::model::ornstein_uhlenbeck
