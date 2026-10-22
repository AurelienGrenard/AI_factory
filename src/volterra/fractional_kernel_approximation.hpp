// Shared CPU/GPU construction of positive exponential approximations to K_H.
#pragma once

#include "volterra/exponential_kernel.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::volterra {
namespace detail {

__host__ __device__ inline double clamp_double(
    double value, double lower, double upper
) {
#if defined(__CUDA_ARCH__)
    return fmin(fmax(value, lower), upper);
#else
    return std::clamp(value, lower, upper);
#endif
}

__host__ __device__ inline double lerp_double(
    double lower, double upper, double fraction
) {
#if defined(__CUDA_ARCH__)
    return lower + fraction * (upper - lower);
#else
    return std::lerp(lower, upper, fraction);
#endif
}

__host__ __device__ inline double regularized_lower_gamma(
    double shape, double x
) {
    if (x <= 0.0) return 0.0;
    constexpr int maximum_iterations = 256;
    constexpr double tolerance = 2.0e-15;
    constexpr double floor = 1.0e-300;
    const double log_scale = -x + shape * std::log(x) - std::lgamma(shape);
    if (x < shape + 1.0) {
        double term = 1.0 / shape;
        double sum = term;
        double denominator = shape;
        for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
            denominator += 1.0;
            term *= x / denominator;
            sum += term;
            if (std::abs(term) <= std::abs(sum) * tolerance) break;
        }
        return clamp_double(sum * std::exp(log_scale), 0.0, 1.0);
    }

    double b = x + 1.0 - shape;
    double c = 1.0 / floor;
    double d = 1.0 / b;
    double fraction = d;
    for (int iteration = 1; iteration <= maximum_iterations; ++iteration) {
        const double i = static_cast<double>(iteration);
        const double coefficient = -i * (i - shape);
        b += 2.0;
        d = coefficient * d + b;
        if (std::abs(d) < floor) d = floor;
        c = b + coefficient / c;
        if (std::abs(c) < floor) c = floor;
        d = 1.0 / d;
        const double delta = d * c;
        fraction *= delta;
        if (std::abs(delta - 1.0) <= tolerance) break;
    }
    const double upper = std::exp(log_scale) * fraction;
    return clamp_double(1.0 - upper, 0.0, 1.0);
}

template<std::size_t FactorCount>
__host__ __device__ ExponentialKernel<FactorCount> positive_geometric_rule(
    double hurst_exponent,
    double log_first_boundary,
    double log_last_boundary
) {
    ExponentialKernel<FactorCount> result{};
    const double measure_exponent = 0.5 - hurst_exponent;
    const double measure_scale = std::cos(
        3.141592653589793238462643383279502884 * hurst_exponent
    ) / 3.141592653589793238462643383279502884;
    double boundaries[FactorCount + 1U]{};
    boundaries[0] = 0.0;
    for (std::size_t boundary = 1U; boundary <= FactorCount; ++boundary) {
        const double interpolation = FactorCount == 1U
            ? 1.0
            : static_cast<double>(boundary - 1U)
                / static_cast<double>(FactorCount - 1U);
        boundaries[boundary] = std::exp(
            log_first_boundary
            + interpolation * (log_last_boundary - log_first_boundary)
        );
    }

    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        const double lower = boundaries[factor];
        const double upper = boundaries[factor + 1U];
        const double lower_measure = std::pow(lower, measure_exponent);
        const double upper_measure = std::pow(upper, measure_exponent);
        const double mass = measure_scale
            * (upper_measure - lower_measure) / measure_exponent;
        const double first_moment = measure_scale
            * (std::pow(upper, measure_exponent + 1.0)
               - std::pow(lower, measure_exponent + 1.0))
            / (measure_exponent + 1.0);
        result.nodes[factor] = static_cast<float>(first_moment / mass);
        result.weights[factor] = static_cast<float>(mass);
    }
    return result;
}

template<std::size_t FactorCount>
__host__ __device__ double squared_l2_error(
    const ExponentialKernel<FactorCount>& approximation,
    double hurst_exponent,
    double horizon,
    double minimum_time = 0.0
) {
    const double shape = hurst_exponent + 0.5;
    const double gamma_shape = std::tgamma(shape);
    double value = (
        std::pow(horizon, 2.0 * hurst_exponent)
        - std::pow(minimum_time, 2.0 * hurst_exponent)
    )
        / (2.0 * hurst_exponent * gamma_shape * gamma_shape);
    for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
        const double node = approximation.nodes[factor];
        const double weight = approximation.weights[factor];
        const double cross = std::pow(node, -shape)
            * (regularized_lower_gamma(shape, node * horizon)
               - regularized_lower_gamma(shape, node * minimum_time));
        value -= 2.0 * weight * cross;
        for (std::size_t other = 0U; other < FactorCount; ++other) {
            const double rate = node + approximation.nodes[other];
            value += weight * approximation.weights[other]
                * (std::exp(-rate * minimum_time)
                   - std::exp(-rate * horizon)) / rate;
        }
    }
    return fmax(value, 0.0);
}

template<std::size_t FactorCount>
__host__ __device__ void optimize_positive_weights(
    ExponentialKernel<FactorCount>& approximation,
    double hurst_exponent,
    double horizon,
    double minimum_time
) {
    const double shape = hurst_exponent + 0.5;
    double weights[FactorCount]{};
    double target[FactorCount]{};
    double gram[FactorCount * FactorCount]{};
    for (std::size_t row = 0U; row < FactorCount; ++row) {
        weights[row] = fmax(
            static_cast<double>(approximation.weights[row]), 1.0e-14
        );
        const double node = approximation.nodes[row];
        target[row] = std::pow(node, -shape)
            * (regularized_lower_gamma(shape, node * horizon)
               - regularized_lower_gamma(shape, node * minimum_time));
        for (std::size_t column = 0U; column < FactorCount; ++column) {
            const double rate = node + approximation.nodes[column];
            gram[row * FactorCount + column] =
                (std::exp(-rate * minimum_time)
                 - std::exp(-rate * horizon)) / rate;
        }
    }

    for (int sweep = 0; sweep < 2048; ++sweep) {
        double maximum_change = 0.0;
        for (std::size_t row = 0U; row < FactorCount; ++row) {
            double residual = target[row];
            for (std::size_t column = 0U; column < FactorCount; ++column) {
                if (column != row) {
                    residual -= gram[row * FactorCount + column]
                        * weights[column];
                }
            }
            const double next = fmax(
                residual / gram[row * FactorCount + row], 1.0e-14
            );
            maximum_change = fmax(
                maximum_change,
                std::abs(next - weights[row])
                    / fmax(weights[row], 1.0e-14)
            );
            weights[row] = next;
        }
        if (maximum_change < 2.0e-12) break;
    }
    for (std::size_t factor = 0U; factor < FactorCount; ++factor)
        approximation.weights[factor] = static_cast<float>(weights[factor]);
}

template<std::size_t FactorCount>
__host__ __device__ ExponentialKernel<FactorCount> optimize_positive_rule(
    ExponentialKernel<FactorCount> approximation,
    double hurst_exponent,
    double horizon,
    double minimum_time,
    double minimum_node,
    double maximum_node
) {
    optimize_positive_weights(
        approximation, hurst_exponent, horizon, minimum_time
    );
    double best_error = squared_l2_error(
        approximation, hurst_exponent, horizon, minimum_time
    );
    double logarithmic_step = 0.8;
    for (int refinement = 0; refinement < 72; ++refinement) {
        bool improved = false;
        for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
            const double current_log = std::log(approximation.nodes[factor]);
            const double lower_log = factor == 0U
                ? std::log(minimum_node)
                : std::log(approximation.nodes[factor - 1U]) + 1.0e-5;
            const double upper_log = factor + 1U == FactorCount
                ? std::log(maximum_node)
                : std::log(approximation.nodes[factor + 1U]) - 1.0e-5;
            ExponentialKernel<FactorCount> best_candidate = approximation;
            double local_error = best_error;
            for (int sign = -1; sign <= 1; sign += 2) {
                const double candidate_log = clamp_double(
                    current_log + static_cast<double>(sign) * logarithmic_step,
                    lower_log,
                    upper_log
                );
                if (candidate_log == current_log) continue;
                ExponentialKernel<FactorCount> candidate = approximation;
                candidate.nodes[factor] = static_cast<float>(
                    std::exp(candidate_log)
                );
                optimize_positive_weights(
                    candidate, hurst_exponent, horizon, minimum_time
                );
                const double error = squared_l2_error(
                    candidate, hurst_exponent, horizon, minimum_time
                );
                if (error + 1.0e-20 < local_error) {
                    local_error = error;
                    best_candidate = candidate;
                }
            }
            if (local_error + 1.0e-20 < best_error) {
                best_error = local_error;
                approximation = best_candidate;
                improved = true;
            }
        }
        if (!improved || refinement % 6 == 5) logarithmic_step *= 0.5;
        if (logarithmic_step < 2.0e-5) break;
    }
    optimize_positive_weights(
        approximation, hurst_exponent, horizon, minimum_time
    );
    return approximation;
}

}  // namespace detail

// Positive bounded-L2 rule. Nodes are barycentres of geometric cells in the
// fractional kernel's Laplace measure; the two truncation bounds are fitted
// deterministically on [smallest_time_scale, horizon], the interval resolved
// by the numerical grid. Positivity is required by the weak
// rough-Heston scheme and is therefore not traded for an unconstrained fit.
// Host validation precedes the shared fit; device callers validate parameter
// bounds before launch, so each thread can run the same numerical fit directly.
template<std::size_t FactorCount>
__host__ __device__ ExponentialKernel<FactorCount>
fit_positive_fractional_kernel_l2_unchecked(
    float hurst_exponent,
    float horizon,
    float smallest_time_scale
) {
    const double h = hurst_exponent;
    const double time_horizon = horizon;
    const double first_min = 0.005 / time_horizon;
    const double first_max = 1.0 / time_horizon;
    const double last_min = 8.0 / time_horizon;
    const double last_max = fmax(
        64.0 / time_horizon,
        24.0 / static_cast<double>(smallest_time_scale)
    );
    double best_first = std::log(first_min);
    double best_last = std::log(last_min);
    double best_error = HUGE_VAL;

    constexpr int coarse_first_count = FactorCount == 1U ? 1 : 9;
    constexpr int coarse_last_count = 13;
    for (int first_index = 0; first_index < coarse_first_count; ++first_index) {
        const double first_fraction = coarse_first_count == 1
            ? 0.0
            : static_cast<double>(first_index)
                / static_cast<double>(coarse_first_count - 1);
        const double log_first = detail::lerp_double(
            std::log(first_min), std::log(first_max), first_fraction
        );
        for (int last_index = 0; last_index < coarse_last_count; ++last_index) {
            const double last_fraction = static_cast<double>(last_index)
                / static_cast<double>(coarse_last_count - 1);
            const double log_last = detail::lerp_double(
                std::log(last_min), std::log(last_max), last_fraction
            );
            if (log_last <= log_first) continue;
            auto candidate = detail::positive_geometric_rule<FactorCount>(
                h, log_first, log_last
            );
            detail::optimize_positive_weights(
                candidate, h, time_horizon, smallest_time_scale
            );
            const double error = detail::squared_l2_error(
                candidate, h, time_horizon, smallest_time_scale
            );
            if (error < best_error) {
                best_error = error;
                best_first = log_first;
                best_last = log_last;
            }
        }
    }

    double first_step = 0.25 * std::log(first_max / first_min);
    double last_step = 0.25 * std::log(last_max / last_min);
    for (int refinement = 0; refinement < 28; ++refinement) {
        bool improved = false;
        for (int first_sign = -1; first_sign <= 1; ++first_sign) {
            for (int last_sign = -1; last_sign <= 1; ++last_sign) {
                const double log_first = detail::clamp_double(
                    best_first + static_cast<double>(first_sign) * first_step,
                    std::log(first_min),
                    std::log(first_max)
                );
                const double log_last = detail::clamp_double(
                    best_last + static_cast<double>(last_sign) * last_step,
                    std::log(last_min),
                    std::log(last_max)
                );
                if (log_last <= log_first) continue;
                auto candidate =
                    detail::positive_geometric_rule<FactorCount>(
                        h, log_first, log_last
                    );
                detail::optimize_positive_weights(
                    candidate, h, time_horizon, smallest_time_scale
                );
                const double error = detail::squared_l2_error(
                    candidate, h, time_horizon, smallest_time_scale
                );
                if (error + 1.0e-18 < best_error) {
                    best_error = error;
                    best_first = log_first;
                    best_last = log_last;
                    improved = true;
                }
            }
        }
        if (!improved) {
            first_step *= 0.5;
            last_step *= 0.5;
        }
    }
    const auto geometric = detail::positive_geometric_rule<FactorCount>(
        h, best_first, best_last
    );
    return detail::optimize_positive_rule(
        geometric,
        h,
        time_horizon,
        smallest_time_scale,
        first_min * 1.0e-3,
        last_max * 4.0
    );
}

template<std::size_t FactorCount>
ExponentialKernel<FactorCount> fit_positive_fractional_kernel_l2(
    float hurst_exponent,
    float horizon,
    float smallest_time_scale
) {
    if (!std::isfinite(hurst_exponent)
        || !(hurst_exponent > 0.0f && hurst_exponent < 0.5f)) {
        throw std::invalid_argument(
            "Fractional-kernel H must be finite and lie in (0, 0.5)."
        );
    }
    if (!std::isfinite(horizon) || !(horizon > 0.0f)
        || !std::isfinite(smallest_time_scale)
        || !(smallest_time_scale > 0.0f)
        || !(smallest_time_scale < horizon)) {
        throw std::invalid_argument(
            "Fractional-kernel time scale must lie strictly inside horizon."
        );
    }

    return fit_positive_fractional_kernel_l2_unchecked<FactorCount>(
        hurst_exponent, horizon, smallest_time_scale
    );
}

template<std::size_t FactorCount>
__device__ ExponentialKernel<FactorCount>
fit_positive_fractional_kernel_l2_device(
    float hurst_exponent,
    float horizon,
    float smallest_time_scale
) {
    return fit_positive_fractional_kernel_l2_unchecked<FactorCount>(
        hurst_exponent, horizon, smallest_time_scale
    );
}

template<std::size_t FactorCount>
double fractional_kernel_relative_l2_error(
    const ExponentialKernel<FactorCount>& approximation,
    float hurst_exponent,
    float horizon,
    float minimum_time = 0.0f
) {
    const double h = hurst_exponent;
    const double gamma_shape = std::tgamma(h + 0.5);
    const double norm = (
        std::pow(static_cast<double>(horizon), 2.0 * h)
        - std::pow(static_cast<double>(minimum_time), 2.0 * h)
    )
        / (2.0 * h * gamma_shape * gamma_shape);
    return std::sqrt(
        detail::squared_l2_error(
            approximation, h, horizon, minimum_time
        ) / norm
    );
}

template<std::size_t FactorCount, std::size_t GridPointCount>
struct PositiveFractionalKernelHurstGrid {
    static_assert(GridPointCount >= 2U);

    float minimum_hurst_exponent = 0.0f;
    float maximum_hurst_exponent = 0.0f;
    std::array<ExponentialKernel<FactorCount>, GridPointCount> kernels{};

    ExponentialKernel<FactorCount> interpolate(float hurst_exponent) const {
        if (!std::isfinite(hurst_exponent)
            || hurst_exponent < minimum_hurst_exponent
            || hurst_exponent > maximum_hurst_exponent) {
            throw std::invalid_argument(
                "Fractional-kernel H lies outside the interpolation grid."
            );
        }
        const double position = std::clamp(
            static_cast<double>(hurst_exponent - minimum_hurst_exponent)
                / static_cast<double>(
                    maximum_hurst_exponent - minimum_hurst_exponent
                )
                * static_cast<double>(GridPointCount - 1U),
            0.0,
            static_cast<double>(GridPointCount - 1U)
        );
        const std::size_t lower = static_cast<std::size_t>(position);
        const std::size_t upper = std::min(
            lower + 1U,
            GridPointCount - 1U
        );
        const float fraction = static_cast<float>(
            position - static_cast<double>(lower)
        );
        ExponentialKernel<FactorCount> result{};
        for (std::size_t factor = 0U; factor < FactorCount; ++factor) {
            result.nodes[factor] = std::lerp(
                kernels[lower].nodes[factor],
                kernels[upper].nodes[factor],
                fraction
            );
            result.weights[factor] = std::lerp(
                kernels[lower].weights[factor],
                kernels[upper].weights[factor],
                fraction
            );
        }
        return result;
    }
};

template<std::size_t FactorCount, std::size_t GridPointCount>
PositiveFractionalKernelHurstGrid<FactorCount, GridPointCount>
fit_positive_fractional_kernel_l2_hurst_grid(
    float minimum_hurst_exponent,
    float maximum_hurst_exponent,
    float horizon,
    float smallest_time_scale
) {
    if (!std::isfinite(minimum_hurst_exponent)
        || !std::isfinite(maximum_hurst_exponent)
        || !(minimum_hurst_exponent > 0.0f)
        || !(minimum_hurst_exponent < maximum_hurst_exponent)
        || !(maximum_hurst_exponent < 0.5f)) {
        throw std::invalid_argument(
            "Fractional-kernel H grid must lie strictly inside (0, 0.5)."
        );
    }
    PositiveFractionalKernelHurstGrid<FactorCount, GridPointCount> result{};
    result.minimum_hurst_exponent = minimum_hurst_exponent;
    result.maximum_hurst_exponent = maximum_hurst_exponent;
    for (std::size_t index = 0U; index < GridPointCount; ++index) {
        const float fraction = static_cast<float>(index)
            / static_cast<float>(GridPointCount - 1U);
        const float hurst_exponent = std::lerp(
            minimum_hurst_exponent,
            maximum_hurst_exponent,
            fraction
        );
        result.kernels[index] = fit_positive_fractional_kernel_l2<
            FactorCount
        >(hurst_exponent, horizon, smallest_time_scale);
    }
    return result;
}

}  // namespace ai_factory::workbench::volterra
