// Stable reusable moments for one Gaussian mean-reverting factor.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench::fixed_income::mean_reverting_gaussian {

// Conditional moments of the future integral of one centered factor.
struct IntegralMoments {
    float state_loading;
    float variance;
};

// Convert a stationary standard deviation into the diffusion volatility.
__host__ __device__ __forceinline__ float volatility_from_stationary_deviation(
    float stationary_standard_deviation,
    float mean_reversion
) {
    return stationary_standard_deviation * sqrtf(2.0f * mean_reversion);
}

// Return the inverse stationary scale used to normalize Gaussian states.
__host__ __device__ __forceinline__ float
inverse_stationary_deviation_from_volatility(
    float volatility,
    float mean_reversion
) {
    return sqrtf(2.0f * mean_reversion) / volatility;
}

namespace detail {

// Evaluate B(delta) by series when mean_reversion * delta is small.
__device__ __forceinline__ float small_time_integral_state_loading(
    float delta,
    float scaled_time
) {
    float normalized = fmaf(scaled_time, 1.0f / 120.0f, -1.0f / 24.0f);
    normalized = fmaf(scaled_time, normalized, 1.0f / 6.0f);
    normalized = fmaf(scaled_time, normalized, -0.5f);
    normalized = fmaf(scaled_time, normalized, 1.0f);
    return delta * normalized;
}

// Avoid the cancellation of O(delta) terms in an O(delta^3) variance.
__device__ __forceinline__ float small_time_integral_variance(
    float volatility_squared,
    float delta,
    float scaled_time
) {
    const float scaled_time2 = scaled_time * scaled_time;
    const float normalized =
        1.0f / 3.0f
        - scaled_time / 4.0f
        + 7.0f * scaled_time2 / 60.0f
        - scaled_time2 * scaled_time / 24.0f;
    return volatility_squared * delta * delta * delta * normalized;
}

}  // namespace detail

// Return B(delta), the loading of the current state in its future integral.
__device__ __forceinline__ float integral_state_loading(
    float mean_reversion,
    float delta
) {
    const float scaled_time = mean_reversion * delta;
    if (fabsf(scaled_time) < 0.02f) {
        return detail::small_time_integral_state_loading(delta, scaled_time);
    }
    return -expm1f(-scaled_time) / mean_reversion;
}

// Reuse caller-owned decay terms without another exponential evaluation.
__device__ __forceinline__ float state_variance_from_decay(
    float mean_reversion,
    float volatility_squared,
    float decay,
    float one_minus_decay
) {
    return volatility_squared
        * one_minus_decay * (1.0f + decay) / (2.0f * mean_reversion);
}

// Return the exact variance of one future factor state.
__device__ __forceinline__ float state_variance(
    float mean_reversion,
    float volatility,
    float delta
) {
    return volatility * volatility
        * (-expm1f(-2.0f * mean_reversion * delta))
        / (2.0f * mean_reversion);
}

// Return Cov(X_next, integral X) from a caller-owned one-minus-decay.
__device__ __forceinline__ float state_integral_covariance_from_decay(
    float mean_reversion,
    float volatility_squared,
    float one_minus_decay
) {
    return volatility_squared * one_minus_decay * one_minus_decay
        / (2.0f * mean_reversion * mean_reversion);
}

// Reuse caller-owned decay terms in the exact integral variance.
__device__ __forceinline__ float integral_variance_from_decay(
    float mean_reversion,
    float volatility_squared,
    float delta,
    float decay,
    float one_minus_decay
) {
    const float scaled_time = mean_reversion * delta;
    if (fabsf(scaled_time) < 0.02f) {
        return detail::small_time_integral_variance(
            volatility_squared, delta, scaled_time
        );
    }

    const float bracket =
        delta
        - 2.0f * one_minus_decay / mean_reversion
        + one_minus_decay * (1.0f + decay)
            / (2.0f * mean_reversion);
    return volatility_squared * bracket
        / (mean_reversion * mean_reversion);
}

// Return the variance of the future integral of one centered factor.
__device__ __forceinline__ float integral_variance(
    float mean_reversion,
    float volatility,
    float delta
) {
    const float scaled_time = mean_reversion * delta;
    const float volatility_squared = volatility * volatility;
    if (fabsf(scaled_time) < 0.02f) {
        return detail::small_time_integral_variance(
            volatility_squared, delta, scaled_time
        );
    }
    const float one_minus_decay = -expm1f(-scaled_time);
    const float decay = 1.0f - one_minus_decay;
    return integral_variance_from_decay(
        mean_reversion,
        volatility_squared,
        delta,
        decay,
        one_minus_decay
    );
}

// Compute loading and variance while sharing the exponential decay.
__device__ __forceinline__ IntegralMoments integral_moments(
    float mean_reversion,
    float volatility,
    float delta
) {
    const float scaled_time = mean_reversion * delta;
    const float volatility_squared = volatility * volatility;
    if (fabsf(scaled_time) < 0.02f) {
        return {
            detail::small_time_integral_state_loading(delta, scaled_time),
            detail::small_time_integral_variance(
                volatility_squared, delta, scaled_time
            ),
        };
    }
    const float one_minus_decay = -expm1f(-scaled_time);
    const float decay = 1.0f - one_minus_decay;
    return {
        one_minus_decay / mean_reversion,
        integral_variance_from_decay(
            mean_reversion,
            volatility_squared,
            delta,
            decay,
            one_minus_decay
        ),
    };
}

}  // namespace ai_factory::workbench::fixed_income::mean_reverting_gaussian
