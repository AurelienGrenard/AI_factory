// Deterministic device probabilities for Gamma and non-central chi-square laws.
#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace ai_factory::workbench {

// Keep both tails together so option pricers never form 1-CDF in FP32.
struct DistributionProbabilities {
    float cdf;
    float survival;
};

namespace noncentral_chi_square_detail {

constexpr double kInverseSqrtTwo = 0.70710678118654752440084436210485;
constexpr double kInverseSqrtTwoPi = 0.39894228040143267793994605993438;
constexpr double kSeriesTolerance = 2.0e-14;
constexpr int kMaximumGammaIterations = 10000;
constexpr int kMaximumPoissonIterations = 10000;
constexpr double kSaddlepointThreshold = 1024.0;

__device__ __forceinline__ double clamp_probability(double value) {
    return fmin(1.0, fmax(0.0, value));
}

// Evaluate P(a,x) directly when x is left of the Gamma transition region.
__device__ __forceinline__ double regularized_gamma_series(
    double shape,
    double value
) {
    double term = 1.0 / shape;
    double sum = term;
    double denominator = shape;
    for (int iteration = 1;
         iteration <= kMaximumGammaIterations;
         ++iteration) {
        denominator += 1.0;
        term *= value / denominator;
        sum += term;
        if (fabs(term) <= fabs(sum) * kSeriesTolerance) break;
    }
    const double log_scale = -value + shape * log(value) - lgamma(shape);
    return clamp_probability(sum * exp(log_scale));
}

// Evaluate Q(a,x) directly through the modified-Lentz continued fraction.
__device__ __forceinline__ double regularized_gamma_continued_fraction(
    double shape,
    double value
) {
    constexpr double tiny = 1.0e-300;
    double b = value + 1.0 - shape;
    double c = 1.0 / tiny;
    double d = 1.0 / fmax(fabs(b), tiny);
    if (b < 0.0) d = -d;
    double fraction = d;
    for (int iteration = 1;
         iteration <= kMaximumGammaIterations;
         ++iteration) {
        const double index = static_cast<double>(iteration);
        const double numerator = -index * (index - shape);
        b += 2.0;
        d = numerator * d + b;
        if (fabs(d) < tiny) d = copysign(tiny, d);
        c = b + numerator / c;
        if (fabs(c) < tiny) c = copysign(tiny, c);
        d = 1.0 / d;
        const double increment = d * c;
        fraction *= increment;
        if (fabs(increment - 1.0) <= kSeriesTolerance) break;
    }
    const double log_scale = -value + shape * log(value) - lgamma(shape);
    return clamp_probability(exp(log_scale) * fraction);
}

__device__ __forceinline__ void regularized_gamma_probabilities_double(
    double shape,
    double value,
    double& cdf,
    double& survival
) {
    if (value <= 0.0) {
        cdf = 0.0;
        survival = 1.0;
        return;
    }
    if (value < shape + 1.0) {
        cdf = regularized_gamma_series(shape, value);
        survival = clamp_probability(1.0 - cdf);
        return;
    }
    survival = regularized_gamma_continued_fraction(shape, value);
    cdf = clamp_probability(1.0 - survival);
}

// Use the Poisson mixture, centered at its modal term, while it remains cheap.
__device__ __forceinline__ DistributionProbabilities poisson_gamma_mixture(
    double degrees_of_freedom,
    double noncentrality,
    double value
) {
    const double gamma_value = 0.5 * value;
    const double gamma_shape = 0.5 * degrees_of_freedom;
    if (noncentrality == 0.0) {
        double cdf = 0.0;
        double survival = 0.0;
        regularized_gamma_probabilities_double(
            gamma_shape, gamma_value, cdf, survival
        );
        return {
            static_cast<float>(cdf),
            static_cast<float>(survival),
        };
    }

    const double poisson_mean = 0.5 * noncentrality;
    const int mode = static_cast<int>(floor(poisson_mean));
    const double mode_as_double = static_cast<double>(mode);
    const double mode_weight = exp(
        -poisson_mean
        + mode_as_double * log(poisson_mean)
        - lgamma(mode_as_double + 1.0)
    );
    double shape = gamma_shape + mode_as_double;
    double gamma_cdf = 0.0;
    double gamma_survival = 0.0;
    regularized_gamma_probabilities_double(
        shape, gamma_value, gamma_cdf, gamma_survival
    );

    // This derivative supplies both stable Gamma-shape recurrences.
    double derivative = exp(
        shape * log(gamma_value) - gamma_value - lgamma(shape + 1.0)
    );
    double cdf_sum = mode_weight * gamma_cdf;
    double survival_sum = mode_weight * gamma_survival;
    double probability_mass = mode_weight;

    // Sum every lower Poisson term; the modal index is at most 512 here.
    double lower_weight = mode_weight;
    double lower_cdf = gamma_cdf;
    double lower_survival = gamma_survival;
    double lower_derivative = derivative;
    double lower_shape = shape;
    for (int index = mode; index > 0; --index) {
        const double previous_derivative = lower_derivative
            * lower_shape / gamma_value;
        lower_cdf = clamp_probability(lower_cdf + previous_derivative);
        lower_survival = clamp_probability(
            lower_survival - previous_derivative
        );
        lower_weight *= static_cast<double>(index) / poisson_mean;
        cdf_sum += lower_weight * lower_cdf;
        survival_sum += lower_weight * lower_survival;
        probability_mass += lower_weight;
        lower_derivative = previous_derivative;
        lower_shape -= 1.0;
    }

    // Sum the infinite upper tail until a geometric bound is negligible.
    double upper_weight = mode_weight;
    double upper_cdf = gamma_cdf;
    double upper_survival = gamma_survival;
    double upper_derivative = derivative;
    double upper_shape = shape;
    for (int index = mode + 1;
         index <= mode + kMaximumPoissonIterations;
         ++index) {
        upper_cdf = clamp_probability(upper_cdf - upper_derivative);
        upper_survival = clamp_probability(
            upper_survival + upper_derivative
        );
        upper_shape += 1.0;
        upper_derivative *= gamma_value / upper_shape;
        upper_weight *= poisson_mean / static_cast<double>(index);
        cdf_sum += upper_weight * upper_cdf;
        survival_sum += upper_weight * upper_survival;
        probability_mass += upper_weight;

        const double next_ratio = poisson_mean
            / static_cast<double>(index + 1);
        if (next_ratio < 1.0) {
            const double remaining_bound = upper_weight * next_ratio
                / (1.0 - next_ratio);
            if (remaining_bound <= kSeriesTolerance) break;
        }
    }

    const double inverse_mass = 1.0 / probability_mass;
    return {
        static_cast<float>(clamp_probability(cdf_sum * inverse_mass)),
        static_cast<float>(
            clamp_probability(survival_sum * inverse_mass)
        ),
    };
}

// Lugannani-Rice is uniformly accurate once the Poisson center is too large.
__device__ __forceinline__ DistributionProbabilities saddlepoint_probabilities(
    double degrees_of_freedom,
    double noncentrality,
    double value
) {
    const double mean = degrees_of_freedom + noncentrality;
    const double variance = 2.0 * (
        degrees_of_freedom + 2.0 * noncentrality
    );
    const double standardized = (value - mean) / sqrt(variance);
    const double skewness = 8.0 * (
        degrees_of_freedom + 3.0 * noncentrality
    ) / (variance * sqrt(variance));

    const double discriminant = sqrt(
        degrees_of_freedom * degrees_of_freedom
        + 4.0 * noncentrality * value
    );
    double y;
    double delta;
    if (fabs(value - mean) <= 0.5 * mean) {
        delta = -2.0 * (value - mean)
            / (2.0 * value - degrees_of_freedom + discriminant);
        y = 1.0 + delta;
    } else {
        y = (degrees_of_freedom + discriminant) / (2.0 * value);
        delta = y - 1.0;
    }

    double cdf;
    double survival;
    if (fabs(delta) <= 1.0e-4) {
        const double density = kInverseSqrtTwoPi
            * exp(-0.5 * standardized * standardized);
        const double correction = skewness
            * (1.0 - standardized * standardized) * density / 6.0;
        cdf = 0.5 * erfc(-standardized * kInverseSqrtTwo)
            + correction;
        survival = 0.5 * erfc(standardized * kInverseSqrtTwo)
            - correction;
    } else {
        const double log_y = fabs(delta) < 0.5
            ? log1p(delta)
            : log(y);
        const double scaled_delta = delta / y;
        const double deviance = degrees_of_freedom
                * (log_y - scaled_delta)
            + noncentrality * scaled_delta * scaled_delta;
        const double signed_root = copysign(
            sqrt(fmax(deviance, 0.0)), -delta
        );
        const double curvature = 2.0 * degrees_of_freedom / (y * y)
            + 4.0 * noncentrality / (y * y * y);
        const double standardized_saddlepoint =
            -0.5 * delta * sqrt(curvature);
        const double density = kInverseSqrtTwoPi
            * exp(-0.5 * signed_root * signed_root);
        const double correction = density * (
            1.0 / signed_root - 1.0 / standardized_saddlepoint
        );
        cdf = 0.5 * erfc(-signed_root * kInverseSqrtTwo) + correction;
        survival = 0.5 * erfc(signed_root * kInverseSqrtTwo) - correction;
    }
    return {
        static_cast<float>(clamp_probability(cdf)),
        static_cast<float>(clamp_probability(survival)),
    };
}

}  // namespace noncentral_chi_square_detail

// Return both regularized incomplete-Gamma tails P(shape,value) and Q.
__device__ __forceinline__ DistributionProbabilities
regularized_gamma_probabilities(float shape, float value) {
    double cdf = 0.0;
    double survival = 0.0;
    noncentral_chi_square_detail::regularized_gamma_probabilities_double(
        static_cast<double>(shape),
        static_cast<double>(value),
        cdf,
        survival
    );
    return {
        static_cast<float>(cdf),
        static_cast<float>(survival),
    };
}

// Return both tails of chi-square(df, noncentrality) at value.
__device__ __forceinline__ DistributionProbabilities
noncentral_chi_square_probabilities(
    float degrees_of_freedom,
    float noncentrality,
    float value
) {
    if (value <= 0.0f) return {0.0f, 1.0f};
    if (static_cast<double>(noncentrality)
        <= noncentral_chi_square_detail::kSaddlepointThreshold) {
        return noncentral_chi_square_detail::poisson_gamma_mixture(
            static_cast<double>(degrees_of_freedom),
            static_cast<double>(noncentrality),
            static_cast<double>(value)
        );
    }
    return noncentral_chi_square_detail::saddlepoint_probabilities(
        static_cast<double>(degrees_of_freedom),
        static_cast<double>(noncentrality),
        static_cast<double>(value)
    );
}

}  // namespace ai_factory::workbench
