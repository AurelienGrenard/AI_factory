// Host implementation of constrained Svensson dataset generation.
#include "tools/datasets/svensson_generation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ai_factory::workbench::datasets::svensson {
namespace {

// Reject empty, reversed, non-finite, or overlapping decay ranges.
void validate_bounds(
    std::size_t row_count,
    const GenerationBounds& bounds
) {
    const auto valid_range = [](const SamplingRange& range) {
        return std::isfinite(range.minimum)
            && std::isfinite(range.maximum)
            && range.minimum <= range.maximum;
    };
    if (row_count == 0U
        || !valid_range(bounds.long_forward)
        || !valid_range(bounds.short_forward)
        || !valid_range(bounds.first_medium_forward)
        || !valid_range(bounds.second_medium_forward)
        || !valid_range(bounds.tau1)
        || !valid_range(bounds.tau2)
        || !valid_range(bounds.accepted_forward)
        || !valid_range(bounds.accepted_curvature)
        || !(bounds.tau1.minimum > 0.0f)
        || !(bounds.tau2.minimum > bounds.tau1.maximum)) {
        throw std::invalid_argument("Invalid Svensson generation bounds.");
    }
}

// Evaluate the analytical instantaneous forward in host precision.
double instantaneous_forward(
    double beta0,
    double beta1,
    double beta2,
    double beta3,
    double tau1,
    double tau2,
    double maturity
) {
    const double x1 = maturity / tau1;
    const double x2 = maturity / tau2;
    return beta0
        + std::exp(-x1) * (beta1 + beta2 * x1)
        + beta3 * x2 * std::exp(-x2);
}

// Reject hidden negative or extreme forwards on a dense maturity grid.
bool has_admissible_forwards(
    float beta0,
    float beta1,
    float beta2,
    float beta3,
    float tau1,
    float tau2,
    const SamplingRange& accepted_forward
) {
    constexpr std::size_t kGridSize = 2'048U;
    constexpr double kMinimumMaturity = 1.0e-4;
    const double maximum_maturity = std::max(100.0, 16.0 * tau2);
    const double log_minimum = std::log(kMinimumMaturity);
    const double log_maximum = std::log(maximum_maturity);
    for (std::size_t index = 0U; index <= kGridSize; ++index) {
        const double weight = static_cast<double>(index) / kGridSize;
        const double maturity = index == 0U
            ? 0.0
            : std::exp(log_minimum + weight * (log_maximum - log_minimum));
        const double forward = instantaneous_forward(
            beta0, beta1, beta2, beta3, tau1, tau2, maturity
        );
        if (forward < accepted_forward.minimum
            || forward > accepted_forward.maximum) {
            return false;
        }
    }
    return true;
}

// Convert one range to stable JSON construction metadata.
nlohmann::ordered_json range_metadata(const SamplingRange& range) {
    constexpr double scale = 10'000'000.0;
    const auto readable = [scale](float value) {
        return std::round(static_cast<double>(value) * scale) / scale;
    };
    return {readable(range.minimum), readable(range.maximum)};
}

}  // namespace

// Fit four forward anchors, then reject implausible intermediate curves.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
) {
    validate_bounds(row_count, bounds);

    std::mt19937_64 generator(seed);
    const auto distribution = [&generator](const SamplingRange& range) {
        return std::uniform_real_distribution<float>(
            range.minimum, range.maximum
        )(generator);
    };

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    const std::size_t maximum_attempts = row_count >
        std::numeric_limits<std::size_t>::max() / 2'000U
        ? std::numeric_limits<std::size_t>::max()
        : row_count * 2'000U;
    for (std::size_t attempt = 0U;
         rows.size() < row_count && attempt < maximum_attempts;
         ++attempt) {
        const float long_forward = distribution(bounds.long_forward);
        const float short_forward = distribution(bounds.short_forward);
        const float first_medium = distribution(bounds.first_medium_forward);
        const float second_medium = distribution(bounds.second_medium_forward);
        const float tau1 = distribution(bounds.tau1);
        const float tau2 = distribution(bounds.tau2);

        const double ratio = static_cast<double>(tau2) / tau1;
        const double inverse_ratio = 1.0 / ratio;
        const double decay_one = std::exp(-1.0);
        const double first_cross = inverse_ratio * std::exp(-inverse_ratio);
        const double second_cross = ratio * std::exp(-ratio);
        const double determinant = decay_one * decay_one
            - first_cross * second_cross;
        const double rhs_first = first_medium - long_forward
            - decay_one * (short_forward - long_forward);
        const double rhs_second = second_medium - long_forward
            - std::exp(-ratio) * (short_forward - long_forward);
        const float beta0 = long_forward;
        const float beta1 = short_forward - long_forward;
        const float beta2 = static_cast<float>(
            (rhs_first * decay_one - first_cross * rhs_second)
            / determinant
        );
        const float beta3 = static_cast<float>(
            (decay_one * rhs_second - second_cross * rhs_first)
            / determinant
        );
        if (!std::isfinite(beta2) || !std::isfinite(beta3)
            || beta2 < bounds.accepted_curvature.minimum
            || beta2 > bounds.accepted_curvature.maximum
            || beta3 < bounds.accepted_curvature.minimum
            || beta3 > bounds.accepted_curvature.maximum
            || !has_admissible_forwards(
                beta0, beta1, beta2, beta3, tau1, tau2,
                bounds.accepted_forward
            )) {
            continue;
        }
        rows.push_back({
            {"beta0", beta0},
            {"beta1", beta1},
            {"beta2", beta2},
            {"beta3", beta3},
            {"tau1", tau1},
            {"tau2", tau2},
        });
    }
    if (rows.size() != row_count) {
        throw std::runtime_error(
            "Svensson bounds rejected too many candidate curves."
        );
    }

    return {
        std::move(rows),
        {
            {"method", "forward-level reconstruction with rejection"},
            {"sampled_factors", {
                {"long_forward", {
                    {"definition", "f(0, infinity)"},
                    {"uniform_bounds", range_metadata(bounds.long_forward)},
                }},
                {"short_forward", {
                    {"definition", "f(0, 0)"},
                    {"uniform_bounds", range_metadata(bounds.short_forward)},
                }},
                {"first_medium_forward", {
                    {"definition", "f(0, tau1)"},
                    {"uniform_bounds", range_metadata(bounds.first_medium_forward)},
                }},
                {"second_medium_forward", {
                    {"definition", "f(0, tau2)"},
                    {"uniform_bounds", range_metadata(bounds.second_medium_forward)},
                }},
                {"tau1", {
                    {"definition", "first decay time in years"},
                    {"uniform_bounds", range_metadata(bounds.tau1)},
                }},
                {"tau2", {
                    {"definition", "second decay time in years"},
                    {"uniform_bounds", range_metadata(bounds.tau2)},
                }},
            }},
            {"acceptance_rule", {
                {"forward_bounds", range_metadata(bounds.accepted_forward)},
                {"curvature_bounds", range_metadata(bounds.accepted_curvature)},
                {"condition", "bounds checked from zero to the converged tail"},
            }},
        },
    };
}

}  // namespace ai_factory::workbench::datasets::svensson
