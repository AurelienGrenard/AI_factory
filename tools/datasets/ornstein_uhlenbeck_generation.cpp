// Host implementation of constrained Ornstein-Uhlenbeck generation.
#include "tools/datasets/ornstein_uhlenbeck_generation.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace ai_factory::workbench::datasets::ornstein_uhlenbeck {
namespace {

// Reject empty, reversed, non-finite, or invalid dynamics bounds.
void validate_dynamics_bounds(
    std::size_t row_count,
    const DynamicsGenerationBounds& bounds
) {
    const auto valid_range = [](const SamplingRange& range) {
        return std::isfinite(range.minimum)
            && std::isfinite(range.maximum)
            && range.minimum <= range.maximum;
    };
    if (row_count == 0U
        || !valid_range(bounds.mean_reversion)
        || !valid_range(bounds.stationary_standard_deviation)
        || !(bounds.mean_reversion.minimum > 0.0f)
        || !(bounds.stationary_standard_deviation.minimum >= 0.0f)) {
        throw std::invalid_argument(
            "Invalid Ornstein-Uhlenbeck generation bounds."
        );
    }
}

// Reject a non-finite or reversed standalone initial-state range.
void validate_initial_state(const SamplingRange& range) {
    if (!std::isfinite(range.minimum)
        || !std::isfinite(range.maximum)
        || range.minimum > range.maximum) {
        throw std::invalid_argument(
            "Invalid Ornstein-Uhlenbeck initial-state bounds."
        );
    }
}

// Convert one range to stable and readable construction metadata.
nlohmann::ordered_json range_metadata(const SamplingRange& range) {
    constexpr double scale = 10'000'000.0;
    const auto readable = [scale](float value) {
        return std::round(static_cast<double>(value) * scale) / scale;
    };
    return {readable(range.minimum), readable(range.maximum)};
}

}  // namespace

// Draw stable reusable OU dynamics without choosing an initial state.
GeneratedRows generate_dynamics_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const DynamicsGenerationBounds& bounds
) {
    validate_dynamics_bounds(row_count, bounds);

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> mean_reversion_distribution(
        bounds.mean_reversion.minimum, bounds.mean_reversion.maximum
    );
    std::uniform_real_distribution<float> stationary_deviation_distribution(
        bounds.stationary_standard_deviation.minimum,
        bounds.stationary_standard_deviation.maximum
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        const float mean_reversion = mean_reversion_distribution(generator);
        const float stationary_standard_deviation =
            stationary_deviation_distribution(generator);
        rows.push_back({
            {"mean_reversion", mean_reversion},
            {
                "volatility",
                stationary_standard_deviation
                    * std::sqrt(2.0f * mean_reversion)
            },
        });
    }

    return {
        std::move(rows),
        {
            {"method", "stationary-dispersion reconstruction"},
            {"sampled_factors", {
                {"mean_reversion", {
                    {"uniform_bounds", range_metadata(bounds.mean_reversion)},
                }},
                {"stationary_standard_deviation", {
                    {
                        "definition",
                        "sigma / sqrt(2 * mean_reversion)"
                    },
                    {
                        "uniform_bounds",
                        range_metadata(
                            bounds.stationary_standard_deviation
                        )
                    },
                }},
            }},
            {"reconstruction", {
                {
                    "volatility",
                    "stationary_standard_deviation * sqrt(2 * mean_reversion)"
                },
            }},
        },
    };
}

// Draw a, x0, and stationary dispersion before reconstructing sigma.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
) {
    validate_dynamics_bounds(row_count, bounds.dynamics);
    validate_initial_state(bounds.initial_state);

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> mean_reversion_distribution(
        bounds.dynamics.mean_reversion.minimum,
        bounds.dynamics.mean_reversion.maximum
    );
    std::uniform_real_distribution<float> initial_state_distribution(
        bounds.initial_state.minimum, bounds.initial_state.maximum
    );
    std::uniform_real_distribution<float> stationary_deviation_distribution(
        bounds.dynamics.stationary_standard_deviation.minimum,
        bounds.dynamics.stationary_standard_deviation.maximum
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        const float mean_reversion = mean_reversion_distribution(generator);
        const float initial_state = initial_state_distribution(generator);
        const float stationary_standard_deviation =
            stationary_deviation_distribution(generator);
        const float volatility = stationary_standard_deviation
            * std::sqrt(2.0f * mean_reversion);
        rows.push_back({
            {"mean_reversion", mean_reversion},
            {"volatility", volatility},
            {"initial_state", initial_state},
        });
    }

    return {
        std::move(rows),
        {
            {"method", "stationary-dispersion reconstruction"},
            {"sampled_factors", {
                {"mean_reversion", {
                    {
                        "uniform_bounds",
                        range_metadata(bounds.dynamics.mean_reversion)
                    },
                }},
                {"initial_state", {
                    {"uniform_bounds", range_metadata(bounds.initial_state)},
                }},
                {"stationary_standard_deviation", {
                    {
                        "definition",
                        "sigma / sqrt(2 * mean_reversion)"
                    },
                    {
                        "uniform_bounds",
                        range_metadata(
                            bounds.dynamics.stationary_standard_deviation
                        )
                    },
                }},
            }},
            {"reconstruction", {
                {
                    "volatility",
                    "stationary_standard_deviation * sqrt(2 * mean_reversion)"
                },
            }},
        },
    };
}

}  // namespace ai_factory::workbench::datasets::ornstein_uhlenbeck
