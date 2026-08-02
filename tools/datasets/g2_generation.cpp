// Host implementation of constrained G2 parameter generation.
#include "tools/datasets/g2_generation.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace ai_factory::workbench::datasets::g2 {
namespace {

// Reject empty, reversed, non-finite, or unstable process bounds.
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
        || !valid_range(bounds.mean_reversion_x)
        || !valid_range(bounds.mean_reversion_gap)
        || !valid_range(bounds.stationary_standard_deviation_x)
        || !valid_range(bounds.stationary_standard_deviation_y)
        || !valid_range(bounds.correlation)
        || !(bounds.mean_reversion_x.minimum > 0.0f)
        || !(bounds.mean_reversion_gap.minimum > 0.0f)
        || !(bounds.stationary_standard_deviation_x.minimum >= 0.0f)
        || !(bounds.stationary_standard_deviation_y.minimum >= 0.0f)
        || bounds.correlation.minimum < -1.0f
        || bounds.correlation.maximum > 1.0f) {
        throw std::invalid_argument("Invalid G2 process generation bounds.");
    }
}

// Reject a non-finite or reversed initial-state range.
void validate_initial_state(const SamplingRange& range) {
    if (!std::isfinite(range.minimum)
        || !std::isfinite(range.maximum)
        || range.minimum > range.maximum) {
        throw std::invalid_argument("Invalid G2 initial-state bounds.");
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

// Draw one common process row and reconstruct sigma and eta.
ParameterRow sample_process_row(
    std::mt19937_64& generator,
    std::uniform_real_distribution<float>& mean_reversion_x_distribution,
    std::uniform_real_distribution<float>& mean_reversion_gap_distribution,
    std::uniform_real_distribution<float>& stationary_x_distribution,
    std::uniform_real_distribution<float>& stationary_y_distribution,
    std::uniform_real_distribution<float>& correlation_distribution
) {
    const float mean_reversion_x =
        mean_reversion_x_distribution(generator);
    const float mean_reversion_y = mean_reversion_x
        + mean_reversion_gap_distribution(generator);
    const float stationary_x = stationary_x_distribution(generator);
    const float stationary_y = stationary_y_distribution(generator);
    return {
        {"mean_reversion_x", mean_reversion_x},
        {
            "volatility_x",
            stationary_x * std::sqrt(2.0f * mean_reversion_x)
        },
        {"mean_reversion_y", mean_reversion_y},
        {
            "volatility_y",
            stationary_y * std::sqrt(2.0f * mean_reversion_y)
        },
        {"correlation", correlation_distribution(generator)},
    };
}

// Describe the constrained sampler once for G2 and G2++ catalogs.
nlohmann::ordered_json process_metadata(
    const ProcessGenerationBounds& bounds
) {
    return {
        {"method", "ordered mean reversions and stationary dispersions"},
        {"sampled_factors", {
            {"mean_reversion_x", {
                {"uniform_bounds", range_metadata(bounds.mean_reversion_x)},
            }},
            {"mean_reversion_gap", {
                {"definition", "mean_reversion_y - mean_reversion_x"},
                {"uniform_bounds", range_metadata(bounds.mean_reversion_gap)},
            }},
            {"stationary_standard_deviation_x", {
                {"definition", "volatility_x / sqrt(2 * mean_reversion_x)"},
                {
                    "uniform_bounds",
                    range_metadata(bounds.stationary_standard_deviation_x)
                },
            }},
            {"stationary_standard_deviation_y", {
                {"definition", "volatility_y / sqrt(2 * mean_reversion_y)"},
                {
                    "uniform_bounds",
                    range_metadata(bounds.stationary_standard_deviation_y)
                },
            }},
            {"correlation", {
                {"uniform_bounds", range_metadata(bounds.correlation)},
            }},
        }},
        {"reconstruction", {
            {"mean_reversion_y", "mean_reversion_x + mean_reversion_gap"},
            {
                "volatility_x",
                "stationary_standard_deviation_x * sqrt(2 * mean_reversion_x)"
            },
            {
                "volatility_y",
                "stationary_standard_deviation_y * sqrt(2 * mean_reversion_y)"
            },
        }},
    };
}

}  // namespace

// Draw stable reusable G2 dynamics without choosing initial states.
GeneratedRows generate_process_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const ProcessGenerationBounds& bounds
) {
    validate_process_bounds(row_count, bounds);
    std::mt19937_64 generator(seed);
    std::uniform_real_distribution<float> mean_reversion_x_distribution(
        bounds.mean_reversion_x.minimum, bounds.mean_reversion_x.maximum
    );
    std::uniform_real_distribution<float> mean_reversion_gap_distribution(
        bounds.mean_reversion_gap.minimum, bounds.mean_reversion_gap.maximum
    );
    std::uniform_real_distribution<float> stationary_x_distribution(
        bounds.stationary_standard_deviation_x.minimum,
        bounds.stationary_standard_deviation_x.maximum
    );
    std::uniform_real_distribution<float> stationary_y_distribution(
        bounds.stationary_standard_deviation_y.minimum,
        bounds.stationary_standard_deviation_y.maximum
    );
    std::uniform_real_distribution<float> correlation_distribution(
        bounds.correlation.minimum, bounds.correlation.maximum
    );

    std::vector<ParameterRow> rows;
    rows.reserve(row_count);
    for (std::size_t row = 0U; row < row_count; ++row) {
        rows.push_back(sample_process_row(
            generator,
            mean_reversion_x_distribution,
            mean_reversion_gap_distribution,
            stationary_x_distribution,
            stationary_y_distribution,
            correlation_distribution
        ));
    }
    return {std::move(rows), process_metadata(bounds)};
}

// Draw G2 process parameters and two bounded initial factor states.
GeneratedRows generate_rows(
    std::size_t row_count,
    std::uint64_t seed,
    const GenerationBounds& bounds
) {
    validate_process_bounds(row_count, bounds.process);
    validate_initial_state(bounds.initial_state_x);
    validate_initial_state(bounds.initial_state_y);

    GeneratedRows generated = generate_process_rows(
        row_count, seed, bounds.process
    );
    std::mt19937_64 generator(seed + 1ULL);
    std::uniform_real_distribution<float> initial_x_distribution(
        bounds.initial_state_x.minimum, bounds.initial_state_x.maximum
    );
    std::uniform_real_distribution<float> initial_y_distribution(
        bounds.initial_state_y.minimum, bounds.initial_state_y.maximum
    );
    for (auto& row : generated.rows) {
        row["initial_state_x"] = initial_x_distribution(generator);
        row["initial_state_y"] = initial_y_distribution(generator);
    }
    generated.construction["sampled_factors"]["initial_state_x"] = {
        {"uniform_bounds", range_metadata(bounds.initial_state_x)},
    };
    generated.construction["sampled_factors"]["initial_state_y"] = {
        {"uniform_bounds", range_metadata(bounds.initial_state_y)},
    };
    return generated;
}

}  // namespace ai_factory::workbench::datasets::g2
