// Deterministic device probabilities for Gamma and non-central chi-square laws.
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ai_factory::workbench {

// Keep both tails together so option pricers never form 1-CDF in FP32.
struct DistributionProbabilities {
    float cdf;
    float survival;
};

namespace noncentral_chi_square_detail {

#if defined(AI_FACTORY_NCX2_FORCE_INLINE)
#define AI_FACTORY_NCX2_LARGE_FUNCTION static __forceinline__
#else
#define AI_FACTORY_NCX2_LARGE_FUNCTION static __noinline__
#endif

constexpr float kInverseSqrtTwo = 0.70710678118654752440f;
constexpr float kInverseSqrtTwoPi = 0.39894228040143267794f;
constexpr float kTwoPi = 6.28318530717958647693f;
// Compensated sums can still collect terms below one ulp of the running sum.
constexpr float kSeriesTolerance = 1.0e-8f;
constexpr float kContinuedFractionTolerance = 1.0e-7f;
constexpr float kLentzTiny = 1.17549435082228750797e-38f;
constexpr std::uint32_t kMaximumGammaIterations = 10000U;
constexpr std::uint32_t kMaximumPoissonIterations = 10000U;
constexpr float kSaddlepointThreshold = 1024.0f;
constexpr float kNearSaddlepointCenter = 1.0e-3f;

__device__ __forceinline__ float clamp_probability(float value) {
    return fminf(1.0f, fmaxf(0.0f, value));
}

// Kahan accumulation retains small positive series terms in FP32.
struct CompensatedSum {
    float sum;
    float correction;

    __device__ __forceinline__ explicit CompensatedSum(float initial)
        : sum(initial), correction(0.0f) {}

    __device__ __forceinline__ void add(float value) {
        const float adjusted = value - correction;
        const float updated = sum + adjusted;
        correction = (updated - sum) - adjusted;
        sum = updated;
    }
};

// Evaluate log(1+x)-x without cancellation around x=0.
__device__ __forceinline__ float log_one_plus_minus_argument(float value) {
    if (fabsf(value) > 0.25f) return log1pf(value) - value;

    float power = value * value;
    CompensatedSum series(-0.5f * power);
    for (std::uint32_t order = 3U; order <= 18U; ++order) {
        power *= value;
        const float sign = (order & 1U) == 0U ? -1.0f : 1.0f;
        series.add(sign * power / static_cast<float>(order));
    }
    return series.sum;
}

// Stirling remainder in log-Gamma, accurate once the shape reaches eight.
__device__ __forceinline__ float stirling_correction(float shape) {
    const float inverse_shape = 1.0f / shape;
    const float inverse_shape_squared = inverse_shape * inverse_shape;
    const float polynomial = fmaf(
        inverse_shape_squared,
        fmaf(
            inverse_shape_squared,
            fmaf(
                inverse_shape_squared,
                -1.0f / 1680.0f,
                1.0f / 1260.0f
            ),
            -1.0f / 360.0f
        ),
        1.0f / 12.0f
    );
    return inverse_shape * polynomial;
}

// Compute -x + a*log(x) - log(Gamma(a)) without large-term cancellation.
__device__ __forceinline__ float gamma_log_scale(
    float shape,
    float value
) {
    if (shape < 8.0f) {
        return -value + shape * logf(value) - lgammaf(shape);
    }

    const float relative_distance = (value - shape) / shape;
    return fmaf(
        shape,
        log_one_plus_minus_argument(relative_distance),
        0.5f * logf(shape / kTwoPi) - stirling_correction(shape)
    );
}

// Evaluate P(a,x) directly when x is left of the Gamma transition region.
__device__ AI_FACTORY_NCX2_LARGE_FUNCTION float regularized_gamma_series(
    float shape,
    float value
) {
    float term = 1.0f / shape;
    CompensatedSum series(term);
    float denominator = shape;
    for (std::uint32_t iteration = 1U;
         iteration <= kMaximumGammaIterations;
         ++iteration) {
        denominator += 1.0f;
        term *= value / denominator;
        series.add(term);
        if (fabsf(term) <= fabsf(series.sum) * kSeriesTolerance) break;
    }
    return clamp_probability(series.sum * expf(gamma_log_scale(shape, value)));
}

// Evaluate Q(a,x) directly through the modified-Lentz continued fraction.
__device__ AI_FACTORY_NCX2_LARGE_FUNCTION float regularized_gamma_continued_fraction(
    float shape,
    float value
) {
    float offset = value + 1.0f - shape;
    float numerator_scale = 1.0f / kLentzTiny;
    float denominator_scale = 1.0f / fmaxf(fabsf(offset), kLentzTiny);
    if (offset < 0.0f) denominator_scale = -denominator_scale;
    float fraction = denominator_scale;

    for (std::uint32_t iteration = 1U;
         iteration <= kMaximumGammaIterations;
         ++iteration) {
        const float index = static_cast<float>(iteration);
        const float numerator = -index * (index - shape);
        offset += 2.0f;

        denominator_scale = fmaf(numerator, denominator_scale, offset);
        if (fabsf(denominator_scale) < kLentzTiny) {
            denominator_scale = copysignf(kLentzTiny, denominator_scale);
        }
        numerator_scale = offset + numerator / numerator_scale;
        if (fabsf(numerator_scale) < kLentzTiny) {
            numerator_scale = copysignf(kLentzTiny, numerator_scale);
        }

        denominator_scale = 1.0f / denominator_scale;
        const float increment = denominator_scale * numerator_scale;
        fraction *= increment;
        if (fabsf(increment - 1.0f) <= kContinuedFractionTolerance) break;
    }

    return clamp_probability(
        expf(gamma_log_scale(shape, value)) * fraction
    );
}

// Select the stable direct Gamma tail, then derive its non-critical complement.
__device__ AI_FACTORY_NCX2_LARGE_FUNCTION DistributionProbabilities
regularized_gamma_probability_pair(float shape, float value) {
    if (value <= 0.0f) return {0.0f, 1.0f};

    if (value < shape + 1.0f) {
        const float cdf = regularized_gamma_series(shape, value);
        return {cdf, clamp_probability(1.0f - cdf)};
    }

    const float survival = regularized_gamma_continued_fraction(shape, value);
    return {clamp_probability(1.0f - survival), survival};
}

// Logarithm of the modal Poisson weight without subtracting large terms.
__device__ __forceinline__ float poisson_mode_log_weight(
    float poisson_mean,
    std::uint32_t mode
) {
    if (mode == 0U) return -poisson_mean;

    const float mode_as_float = static_cast<float>(mode);
    if (mode < 8U) {
        return -poisson_mean
            + mode_as_float * logf(poisson_mean)
            - lgammaf(mode_as_float + 1.0f);
    }

    const float relative_distance =
        (poisson_mean - mode_as_float) / mode_as_float;
    return fmaf(
        mode_as_float,
        log_one_plus_minus_argument(relative_distance),
        -0.5f * logf(kTwoPi * mode_as_float)
            - stirling_correction(mode_as_float)
    );
}

// Use the exact Poisson-Gamma mixture, centered at its modal Poisson term.
__device__ AI_FACTORY_NCX2_LARGE_FUNCTION DistributionProbabilities
poisson_gamma_mixture(
    float degrees_of_freedom,
    float noncentrality,
    float value
) {
    const float gamma_value = 0.5f * value;
    const float gamma_shape = 0.5f * degrees_of_freedom;
    if (noncentrality == 0.0f) {
        return regularized_gamma_probability_pair(gamma_shape, gamma_value);
    }

    const float poisson_mean = 0.5f * noncentrality;
    const std::uint32_t mode =
        static_cast<std::uint32_t>(floorf(poisson_mean));
    const float mode_as_float = static_cast<float>(mode);
    const float mode_weight = expf(
        poisson_mode_log_weight(poisson_mean, mode)
    );
    const float shape = gamma_shape + mode_as_float;
    const DistributionProbabilities gamma_probabilities =
        regularized_gamma_probability_pair(shape, gamma_value);

    // x^a exp(-x) / Gamma(a+1) drives both Gamma-shape recurrences.
    const float derivative = expf(
        gamma_log_scale(shape, gamma_value) - logf(shape)
    );
    CompensatedSum cdf_sum(mode_weight * gamma_probabilities.cdf);
    CompensatedSum survival_sum(
        mode_weight * gamma_probabilities.survival
    );
    CompensatedSum probability_mass(mode_weight);

    // Sum every lower Poisson term; the modal index is at most 512 here.
    float lower_weight = mode_weight;
    float lower_cdf = gamma_probabilities.cdf;
    float lower_survival = gamma_probabilities.survival;
    float lower_derivative = derivative;
    float lower_shape = shape;
    for (std::uint32_t index = mode; index > 0U; --index) {
        const float previous_derivative =
            lower_derivative * lower_shape / gamma_value;
        lower_cdf = clamp_probability(lower_cdf + previous_derivative);
        lower_survival = clamp_probability(
            lower_survival - previous_derivative
        );
        lower_weight *= static_cast<float>(index) / poisson_mean;
        cdf_sum.add(lower_weight * lower_cdf);
        survival_sum.add(lower_weight * lower_survival);
        probability_mass.add(lower_weight);
        lower_derivative = previous_derivative;
        lower_shape -= 1.0f;
    }

    // Sum the infinite upper tail until its geometric bound is negligible.
    float upper_weight = mode_weight;
    float upper_cdf = gamma_probabilities.cdf;
    float upper_survival = gamma_probabilities.survival;
    float upper_derivative = derivative;
    float upper_shape = shape;
    for (std::uint32_t offset = 1U;
         offset <= kMaximumPoissonIterations;
         ++offset) {
        const std::uint32_t index = mode + offset;
        upper_cdf = clamp_probability(upper_cdf - upper_derivative);
        upper_survival = clamp_probability(
            upper_survival + upper_derivative
        );
        upper_shape += 1.0f;
        upper_derivative *= gamma_value / upper_shape;
        upper_weight *= poisson_mean / static_cast<float>(index);
        cdf_sum.add(upper_weight * upper_cdf);
        survival_sum.add(upper_weight * upper_survival);
        probability_mass.add(upper_weight);

        const float next_ratio = poisson_mean
            / static_cast<float>(index + 1U);
        if (next_ratio < 1.0f) {
            const float remaining_bound = upper_weight * next_ratio
                / (1.0f - next_ratio);
            if (remaining_bound
                <= kSeriesTolerance * probability_mass.sum) {
                break;
            }
        }
    }

    const float inverse_mass = 1.0f / probability_mass.sum;
    return {
        clamp_probability(cdf_sum.sum * inverse_mass),
        clamp_probability(survival_sum.sum * inverse_mass),
    };
}

// Evaluate log(1+d)-d/(1+d) stably for the saddlepoint deviance.
__device__ __forceinline__ float log_one_plus_minus_ratio(float delta) {
    return log_one_plus_minus_argument(delta)
        + delta * delta / (1.0f + delta);
}

// Lugannani-Rice remains cheap when the exact Poisson center is too large.
__device__ AI_FACTORY_NCX2_LARGE_FUNCTION DistributionProbabilities
saddlepoint_probabilities(
    float degrees_of_freedom,
    float noncentrality,
    float value
) {
    // This ordering retains a small degrees-of-freedom correction next to a
    // large, nearly equal value and noncentrality.
    const float centered_value =
        (value - noncentrality) - degrees_of_freedom;
    const float mean = degrees_of_freedom + noncentrality;
    const float variance = 2.0f * (
        degrees_of_freedom + 2.0f * noncentrality
    );
    const float standard_deviation = sqrtf(variance);
    const float standardized = centered_value / standard_deviation;
    const float skewness = 8.0f * (
        degrees_of_freedom + 3.0f * noncentrality
    ) / (variance * standard_deviation);

    const float discriminant = sqrtf(fmaf(
        4.0f * noncentrality,
        value,
        degrees_of_freedom * degrees_of_freedom
    ));
    float transformed;
    float delta;
    if (fabsf(centered_value) <= 0.5f * mean) {
        delta = -2.0f * centered_value
            / (2.0f * value - degrees_of_freedom + discriminant);
        transformed = 1.0f + delta;
    } else {
        transformed = (degrees_of_freedom + discriminant)
            / (2.0f * value);
        delta = transformed - 1.0f;
    }

    float cdf;
    float survival;
    if (fabsf(delta) <= kNearSaddlepointCenter) {
        const float density = kInverseSqrtTwoPi
            * expf(-0.5f * standardized * standardized);
        const float correction = skewness
            * (1.0f - standardized * standardized) * density / 6.0f;
        cdf = 0.5f * erfcf(-standardized * kInverseSqrtTwo)
            + correction;
        survival = 0.5f * erfcf(standardized * kInverseSqrtTwo)
            - correction;
    } else {
        const float scaled_delta = delta / transformed;
        const float deviance = fmaf(
            noncentrality,
            scaled_delta * scaled_delta,
            degrees_of_freedom * log_one_plus_minus_ratio(delta)
        );
        const float signed_root = copysignf(
            sqrtf(fmaxf(deviance, 0.0f)), -delta
        );
        const float transformed_squared = transformed * transformed;
        const float curvature =
            2.0f * degrees_of_freedom / transformed_squared
            + 4.0f * noncentrality
                / (transformed_squared * transformed);
        const float standardized_saddlepoint =
            -0.5f * delta * sqrtf(curvature);
        const float density = kInverseSqrtTwoPi
            * expf(-0.5f * signed_root * signed_root);
        const float correction = density * (
            (standardized_saddlepoint - signed_root)
            / (signed_root * standardized_saddlepoint)
        );
        cdf = 0.5f * erfcf(-signed_root * kInverseSqrtTwo) + correction;
        survival = 0.5f * erfcf(signed_root * kInverseSqrtTwo) - correction;
    }

    return {
        clamp_probability(cdf),
        clamp_probability(survival),
    };
}

}  // namespace noncentral_chi_square_detail

#undef AI_FACTORY_NCX2_LARGE_FUNCTION

// Return both regularized incomplete-Gamma tails P(shape,value) and Q.
__device__ __forceinline__ DistributionProbabilities
regularized_gamma_probabilities(float shape, float value) {
    return noncentral_chi_square_detail::regularized_gamma_probability_pair(
        shape, value
    );
}

// Return both tails of chi-square(df, noncentrality) at value.
__device__ __forceinline__ DistributionProbabilities
noncentral_chi_square_probabilities(
    float degrees_of_freedom,
    float noncentrality,
    float value
) {
    if (value <= 0.0f) return {0.0f, 1.0f};
    if (noncentrality <= noncentral_chi_square_detail::kSaddlepointThreshold) {
        return noncentral_chi_square_detail::poisson_gamma_mixture(
            degrees_of_freedom, noncentrality, value
        );
    }
    return noncentral_chi_square_detail::saddlepoint_probabilities(
        degrees_of_freedom, noncentrality, value
    );
}

}  // namespace ai_factory::workbench
