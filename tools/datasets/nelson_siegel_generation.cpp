// Host implementation of constrained Nelson-Siegel dataset generation.
#include "tools/datasets/nelson_siegel_generation.hpp"

#include "curve/nelson_siegel/instantaneous_forward.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ai_factory::workbench::datasets::nelson_siegel {
namespace {

// Reject empty, reversed, non-finite, or non-positive decay ranges.
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
        || !valid_range(bounds.medium_forward)
        || !valid_range(bounds.tau)
        || !valid_range(bounds.accepted_forward)
        || !(bounds.tau.minimum > 0.0f)) {
        throw std::invalid_argument(
            "Invalid Nelson-Siegel generation bounds."
        );
    }
}

// Check both endpoints and the curve's only possible interior extremum.
bool has_admissible_forwards(
    float beta0,
    float beta1,
    float beta2,
    const SamplingRange& accepted_forward
) {
    double minimum = std::min(
        static_cast<double>(beta0),
        static_cast<double>(beta0 + beta1)
    );
    double maximum = std::max(
        static_cast<double>(beta0),
        static_cast<double>(beta0 + beta1)
    );
    if (beta2 != 0.0f) {
        const double critical_scaled_maturity =
            1.0 - static_cast<double>(beta1) / static_cast<double>(beta2);
        if (critical_scaled_maturity > 0.0) {
            const double critical_forward =
                curve::nelson_siegel::instantaneous_forward_formula(
                    static_cast<double>(beta0),
                    static_cast<double>(beta1),
                    static_cast<double>(beta2),
                    critical_scaled_maturity
                );
            minimum = std::min(minimum, critical_forward);
            maximum = std::max(maximum, critical_forward);
        }
    }
    return minimum >= static_cast<double>(accepted_forward.minimum)
        && maximum <= static_cast<double>(accepted_forward.maximum);
}

// Convert one range to stable JSON construction metadata.
nlohmann::ordered_json range_metadata(
    const SamplingRange& range
) {
    constexpr double scale = 10'000'000.0;
    const auto readable = [scale](float value) {
        return std::round(static_cast<double>(value) * scale) / scale;
    };
    return {
        readable(range.minimum),
        readable(range.maximum),
    };
}

}  // namespace

// Sample short, medium, and long forwards, then reconstruct Nelson-Siegel.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
) {
    validate_bounds(row_count, bounds);

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> long_distribution(
        bounds.long_forward.minimum, bounds.long_forward.maximum
    );
    std::uniform_real_distribution<float> short_distribution(
        bounds.short_forward.minimum, bounds.short_forward.maximum
    );
    std::uniform_real_distribution<float> medium_distribution(
        bounds.medium_forward.minimum, bounds.medium_forward.maximum
    );
    std::uniform_real_distribution<float> tau_distribution(
        bounds.tau.minimum, bounds.tau.maximum
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    const std::size_t maximum_attempts =
        row_count > std::numeric_limits<std::size_t>::max() / 1'000U
        ? std::numeric_limits<std::size_t>::max()
        : row_count * 1'000U;
    for (std::size_t attempt = 0U;
         rows.size() < row_count && attempt < maximum_attempts;
         ++attempt) {
        const float long_forward = long_distribution(generator);
        const float short_forward = short_distribution(generator);
        const float medium_forward = medium_distribution(generator);
        const float tau = tau_distribution(generator);

        const float beta0 = long_forward;
        const float beta1 = short_forward - long_forward;
        const float beta2 =
            std::exp(1.0f) * (medium_forward - long_forward) - beta1;
        if (!has_admissible_forwards(
                beta0, beta1, beta2, bounds.accepted_forward
            )) {
            continue;
        }

        rows.push_back({
            {"beta0", beta0},
            {"beta1", beta1},
            {"beta2", beta2},
            {"tau", tau},
        });
    }
    if (rows.size() != row_count) {
        throw std::runtime_error(
            "Nelson-Siegel bounds rejected too many candidate curves."
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
                {"medium_forward", {
                    {"definition", "f(0, tau)"},
                    {"uniform_bounds", range_metadata(bounds.medium_forward)},
                }},
                {"tau", {
                    {"definition", "decay time in years"},
                    {"uniform_bounds", range_metadata(bounds.tau)},
                }},
            }},
            {"reconstruction", {
                {"beta0", "long_forward"},
                {"beta1", "short_forward - long_forward"},
                {
                    "beta2",
                    "e * (medium_forward - long_forward) - beta1"
                },
            }},
            {"acceptance_rule", {
                {
                    "forward_bounds",
                    range_metadata(bounds.accepted_forward)
                },
                {"condition", "bounds apply to f(0,T) for every T >= 0"},
                {
                    "extremum",
                    "T / tau = 1 - beta1 / beta2 when positive"
                },
            }},
        },
    };
}

}  // namespace ai_factory::workbench::datasets::nelson_siegel
