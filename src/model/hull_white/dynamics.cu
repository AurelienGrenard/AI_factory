// Exact CUDA transitions and zero-coupon pricing for Hull-White one-factor.
#include "model/hull_white/dynamics.cuh"

// Include curve formulas so NVCC can inline them into pricing kernels.
#include "curve/nelson_siegel.cu"

#include <cuda_runtime.h>

namespace ai_factory::workbench::hull_white {
namespace {

constexpr std::uint64_t kUniformsPerStep = 2ULL;

// Return B(delta) = (1 - exp(-a delta)) / a.
__device__ __forceinline__ float hull_white_bond_loading(
    float mean_reversion,
    float delta
) {
    return -expm1f(-mean_reversion * delta) / mean_reversion;
}

// Return the variance of the future integral of the Gaussian factor.
__device__ __forceinline__ float hull_white_integral_variance(
    const HullWhiteOneFactorParameters& parameters,
    float delta
) {
    const float a = parameters.mean_reversion;
    const float scaled_time = a * delta;
    if (fabsf(scaled_time) < 0.02f) {
        const float scaled_time2 = scaled_time * scaled_time;
        const float normalized =
            1.0f / 3.0f
            - scaled_time / 4.0f
            + 7.0f * scaled_time2 / 60.0f
            - scaled_time2 * scaled_time / 24.0f;
        return parameters.volatility * parameters.volatility
            * delta * delta * delta * normalized;
    }

    const float one_minus_decay = -expm1f(-a * delta);
    const float one_minus_decay_squared = -expm1f(-2.0f * a * delta);
    const float bracket =
        delta
        - 2.0f * one_minus_decay / a
        + one_minus_decay_squared / (2.0f * a);
    return parameters.volatility * parameters.volatility
        * bracket / (a * a);
}

// Integrate the deterministic shift phi between two model times.
__device__ __forceinline__ float hull_white_shift_integral(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    float start,
    float end
) {
    const float a = parameters.mean_reversion;
    const float delta = end - start;
    const float one_minus_decay = -expm1f(-a * delta);
    const float one_minus_decay_squared = -expm1f(-2.0f * a * delta);
    const float forward_integral =
        curve::nelson_siegel_log_discount(initial_curve, start)
        - curve::nelson_siegel_log_discount(initial_curve, end);
    if (start == 0.0f) {
        return forward_integral
            + 0.5f * hull_white_integral_variance(parameters, end);
    }
    const float convexity_integral =
        parameters.volatility * parameters.volatility / (2.0f * a * a)
        * (
            delta
            - 2.0f * expf(-a * start) * one_minus_decay / a
            + expf(-2.0f * a * start)
                * one_minus_decay_squared / (2.0f * a)
        );
    return forward_integral + convexity_integral;
}

}  // namespace

// Prepare one exact joint Gaussian transition for x and its time integral.
__device__ __forceinline__ HullWhiteStepParameters prepare_hull_white_step(
    const HullWhiteOneFactorParameters& parameters,
    float dt
) {
    const float a = parameters.mean_reversion;
    const float sigma2 = parameters.volatility * parameters.volatility;
    const float decay = expf(-a * dt);
    const float one_minus_decay = -expm1f(-a * dt);
    const float factor_variance =
        sigma2 * (-expm1f(-2.0f * a * dt)) / (2.0f * a);
    const float factor_standard_deviation = sqrtf(factor_variance);
    const float covariance =
        sigma2 * one_minus_decay * one_minus_decay / (2.0f * a * a);
    const float integral_variance =
        hull_white_integral_variance(parameters, dt);
    const float integral_factor_normal_loading =
        covariance / factor_standard_deviation;
    const float independent_variance = fmaxf(
        integral_variance
            - integral_factor_normal_loading
                * integral_factor_normal_loading,
        0.0f
    );

    return {
        dt,
        decay,
        factor_standard_deviation,
        one_minus_decay / a,
        integral_factor_normal_loading,
        sqrtf(independent_variance),
    };
}

// Start from x(0) = 0 and an undiscounted path value of one.
__device__ __forceinline__ HullWhiteState initial_hull_white_state() {
    return {0.0f, 0.0f};
}

// Evaluate the deterministic shift that fits the initial forward curve.
__device__ __forceinline__ float hull_white_short_rate_shift(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    float time
) {
    const float a = parameters.mean_reversion;
    const float one_minus_decay = -expm1f(-a * time);
    const float correction =
        parameters.volatility * parameters.volatility
        * one_minus_decay * one_minus_decay / (2.0f * a * a);
    return curve::nelson_siegel_instantaneous_forward(initial_curve, time)
        + correction;
}

// Add the current Gaussian factor to the deterministic curve shift.
__device__ __forceinline__ float hull_white_short_rate(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float time
) {
    return state.factor
        + hull_white_short_rate_shift(parameters, initial_curve, time);
}

// Combine the stochastic integral with the analytical curve shift.
__device__ __forceinline__ float hull_white_log_discount(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float time
) {
    return -state.integrated_factor
        - hull_white_shift_integral(
            parameters, initial_curve, 0.0f, time
        );
}

// Exponentiate the exact path log-discount only when a payoff needs it.
__device__ __forceinline__ float hull_white_discount_factor(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float time
) {
    return expf(
        hull_white_log_discount(parameters, initial_curve, state, time)
    );
}

// Evaluate the equivalent deterministic drift of the short-rate SDE.
__device__ __forceinline__ float hull_white_drift_level(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    float time
) {
    const float a = parameters.mean_reversion;
    const float correction =
        parameters.volatility * parameters.volatility
        * (-expm1f(-2.0f * a * time)) / (2.0f * a);
    return curve::nelson_siegel_forward_derivative(initial_curve, time)
        + a * curve::nelson_siegel_instantaneous_forward(initial_curve, time)
        + correction;
}

// Simulate x and integral(x ds) jointly without evaluating the curve per step.
__device__ __forceinline__ void one_step_hull_white_transition(
    const HullWhiteStepParameters& step,
    float factor_normal,
    float integral_normal,
    HullWhiteState& state
) {
    const float previous_factor = state.factor;
    const float factor_noise =
        step.factor_standard_deviation * factor_normal;
    const float integral_noise =
        step.integral_factor_normal_loading * factor_normal
        + step.integral_independent_standard_deviation * integral_normal;

    state.factor = step.decay * previous_factor + factor_noise;
    state.integrated_factor +=
        step.integral_factor_loading * previous_factor
        + integral_noise;
}

// Generate all random variates for one terminal fixed-income path.
__device__ __forceinline__ HullWhiteState simulate_terminal_hull_white_state(
    const HullWhiteStepParameters& step,
    philox::PhiloxKey key,
    std::size_t path,
    std::size_t num_steps
) {
    HullWhiteState state = initial_hull_white_state();
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
        one_step_hull_white_transition(
            step,
            normals.first,
            normals.second,
            state
        );
    }
    return state;
}

// Write pre-maturity states in a date-major exercise grid.
__device__ __forceinline__ HullWhiteState
simulate_factor_discount_on_regular_grid(
    const HullWhiteStepParameters& initial_stub_step,
    const HullWhiteStepParameters& regular_step,
    philox::PhiloxKey key,
    std::size_t path,
    std::uint32_t initial_stub_steps,
    std::uint32_t steps_per_exercise,
    std::uint32_t exercise_count,
    std::size_t path_count,
    float* __restrict__ observed_factors,
    float* __restrict__ observed_integrated_factors
) {
    HullWhiteState state = initial_hull_white_state();
    const std::uint64_t total_steps =
        static_cast<std::uint64_t>(initial_stub_steps)
        + static_cast<std::uint64_t>(exercise_count - 1U)
            * static_cast<std::uint64_t>(steps_per_exercise);
    const std::uint64_t uniform_count = kUniformsPerStep * total_steps;
    const std::uint64_t groups_per_path = (uniform_count + 3ULL) >> 2U;
    const std::uint64_t first_group =
        static_cast<std::uint64_t>(path) * groups_per_path;
    philox::UniformSequence uniforms(key, first_group);

    for (std::uint32_t step_index = 0U;
         step_index < initial_stub_steps;
         ++step_index) {
        const float angle_uniform = uniforms.next();
        const float radius_uniform = uniforms.next();
        const philox::NormalPair normals =
            philox::box_muller(angle_uniform, radius_uniform);
        one_step_hull_white_transition(
            initial_stub_step,
            normals.first,
            normals.second,
            state
        );
    }
    if (exercise_count > 1U) {
        observed_factors[path] = state.factor;
        observed_integrated_factors[path] = state.integrated_factor;
    }

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
            one_step_hull_white_transition(
                regular_step,
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

// Price one zero-coupon from the conditional Gaussian factor integral.
__device__ __forceinline__ float hull_white_zero_coupon_bond(
    const HullWhiteOneFactorParameters& parameters,
    const curve::NelsonSiegelParameters& initial_curve,
    const HullWhiteState& state,
    float valuation_time,
    float maturity
) {
    const float delta = maturity - valuation_time;
    const float loading = hull_white_bond_loading(
        parameters.mean_reversion, delta
    );
    const float deterministic_integral = hull_white_shift_integral(
        parameters, initial_curve, valuation_time, maturity
    );
    const float integral_variance =
        hull_white_integral_variance(parameters, delta);
    return expf(
        -loading * state.factor
        - deterministic_integral
        + 0.5f * integral_variance
    );
}

}  // namespace ai_factory::workbench::hull_white
