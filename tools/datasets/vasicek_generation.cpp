// Host implementation of constrained Vasicek generation.
#include "tools/datasets/vasicek_generation.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace ai_factory::workbench::datasets::vasicek {
namespace {

// Reject empty, reversed, non-finite, or invalid process bounds.
void validate_process_bounds(
    std::size_t row_count,
    const ProcessGenerationBounds& bounds
) {
    const auto valid_range = [](const SamplingRange& range) {
        return std::isfinite(range.minimum)
            && std::isfinite(range.maximum)
            && range.minimum <= range.maximum;
    };
    if (row_count == 0U
        || !valid_range(bounds.mean_reversion)
        || !valid_range(bounds.long_term_mean)
        || !valid_range(bounds.stationary_standard_deviation)
        || !(bounds.mean_reversion.minimum > 0.0f)
        || !(bounds.stationary_standard_deviation.minimum >= 0.0f)) {
        throw std::invalid_argument(
            "Invalid Vasicek generation bounds."
        );
    }
}

// Reject a non-finite or reversed standalone initial-state range.
void validate_initial_state(const SamplingRange& range) {
    if (!std::isfinite(range.minimum)
        || !std::isfinite(range.maximum)
        || range.minimum > range.maximum) {
        throw std::invalid_argument(
            "Invalid Vasicek initial-state bounds."
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

// Draw stable reusable Vasicek process parameters without choosing an initial state.
GeneratedRows generate_process_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const ProcessGenerationBounds& bounds
) {
    validate_process_bounds(row_count, bounds);

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> mean_reversion_distribution(
        bounds.mean_reversion.minimum, bounds.mean_reversion.maximum
    );
    std::uniform_real_distribution<float> long_term_mean_distribution(
        bounds.long_term_mean.minimum, bounds.long_term_mean.maximum
    );
    std::uniform_real_distribution<float> stationary_deviation_distribution(
        bounds.stationary_standard_deviation.minimum,
        bounds.stationary_standard_deviation.maximum
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        const float mean_reversion = mean_reversion_distribution(generator);
        const float long_term_mean = long_term_mean_distribution(generator);
        const float stationary_standard_deviation =
            stationary_deviation_distribution(generator);
        rows.push_back({
            {"mean_reversion", mean_reversion},
            {"long_term_mean", long_term_mean},
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
                {"long_term_mean", {
                    {"uniform_bounds", range_metadata(bounds.long_term_mean)},
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

// Draw a, b, r0, and stationary dispersion before reconstructing sigma.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
) {
    validate_process_bounds(row_count, bounds.process);
    validate_initial_state(bounds.initial_state);

    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> mean_reversion_distribution(
        bounds.process.mean_reversion.minimum,
        bounds.process.mean_reversion.maximum
    );
    std::uniform_real_distribution<float> long_term_mean_distribution(
        bounds.process.long_term_mean.minimum,
        bounds.process.long_term_mean.maximum
    );
    std::uniform_real_distribution<float> initial_state_distribution(
        bounds.initial_state.minimum, bounds.initial_state.maximum
    );
    std::uniform_real_distribution<float> stationary_deviation_distribution(
        bounds.process.stationary_standard_deviation.minimum,
        bounds.process.stationary_standard_deviation.maximum
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        const float mean_reversion = mean_reversion_distribution(generator);
        const float long_term_mean = long_term_mean_distribution(generator);
        const float initial_state = initial_state_distribution(generator);
        const float stationary_standard_deviation =
            stationary_deviation_distribution(generator);
        const float volatility = stationary_standard_deviation
            * std::sqrt(2.0f * mean_reversion);
        rows.push_back({
            {"mean_reversion", mean_reversion},
            {"long_term_mean", long_term_mean},
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
                        range_metadata(bounds.process.mean_reversion)
                    },
                }},
                {"long_term_mean", {
                    {
                        "uniform_bounds",
                        range_metadata(bounds.process.long_term_mean)
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
                            bounds.process.stationary_standard_deviation
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

}  // namespace ai_factory::workbench::datasets::vasicek
