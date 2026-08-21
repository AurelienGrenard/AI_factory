// Host implementation of constrained core-tail Cliquet generation.
#include "tools/datasets/cliquet_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace ai_factory::workbench::datasets::cliquet {
namespace {

constexpr std::array<std::uint32_t, 2U> kCoreIntervals = {
    21U, 63U,
};
constexpr std::array<std::uint32_t, 3U> kTailIntervals = {
    5U, 126U, 252U,
};

// Draw one uniform FP32 value without exposing RNG details to generators.
float uniform(
    std::mt19937_64& generator,
    float minimum,
    float maximum
) {
    return std::uniform_real_distribution<float>(minimum, maximum)(generator);
}

// Draw an issuance schedule and return maturity, interval, and date count.
template <std::size_t IntervalCount>
std::tuple<std::uint32_t, std::uint32_t, std::size_t> schedule(
    std::mt19937_64& generator,
    const std::array<std::uint32_t, IntervalCount>& intervals,
    std::uint32_t minimum_maturity,
    std::uint32_t maximum_maturity
) {
    const std::uint32_t interval = intervals[
        std::uniform_int_distribution<std::size_t>(
            0U, intervals.size() - 1U
        )(generator)
    ];
    const std::size_t minimum_observations = std::max<std::size_t>(
        1U, (minimum_maturity + interval - 1U) / interval
    );
    const std::size_t maximum_observations = maximum_maturity / interval;
    const std::size_t observation_count =
        std::uniform_int_distribution<std::size_t>(
            minimum_observations, maximum_observations
        )(generator);
    return {
        interval * static_cast<std::uint32_t>(observation_count),
        interval,
        observation_count,
    };
}

// Draw a meaningful global cap below the theoretical local-cap sum.
float global_cap(
    std::mt19937_64& generator,
    float global_floor,
    float maximum_aggregate,
    float lower_fraction,
    float upper_fraction,
    float absolute_maximum
) {
    const float bounded_maximum = std::min(
        maximum_aggregate, absolute_maximum
    );
    const float minimum_gap = std::max(0.001f, 0.05f * bounded_maximum);
    const float minimum = std::max(
        global_floor + minimum_gap,
        lower_fraction * bounded_maximum
    );
    const float maximum = std::max(
        minimum,
        upper_fraction * bounded_maximum
    );
    return uniform(generator, minimum, maximum);
}

// Reject empty regimes before constructing their distributions.
void validate_counts(
    std::size_t core_row_count,
    std::size_t tail_row_count
) {
    if (core_row_count == 0U || tail_row_count == 0U) {
        throw std::invalid_argument(
            "Cliquet generation requires non-empty core and stress regimes."
        );
    }
}

}  // namespace

// Generate constrained periodic-return terms in traceable regime order.
GeneratedRows generate_rows(
    std::size_t core_row_count,
    std::size_t tail_row_count,
    std::uint64_t seed
) {
    validate_counts(core_row_count, tail_row_count);
    std::mt19937_64 generator(seed);
    GeneratedRows generated;
    generated.rows.reserve(core_row_count + tail_row_count);

    for (std::size_t row = 0U; row < core_row_count; ++row) {
        const auto [maturity, interval, observation_count] = schedule(
            generator, kCoreIntervals, 252U, 1260U
        );
        const float local_floor = uniform(generator, -0.05f, 0.0f);
        const float local_cap = uniform(generator, 0.01f, 0.08f);
        const float maximum_aggregate =
            static_cast<float>(observation_count) * local_cap;
        const float global_floor = uniform(
            generator, 0.0f, std::min(0.03f, 0.15f * maximum_aggregate)
        );
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {"participation_rate", uniform(generator, 0.80f, 1.20f)},
            {"local_floor", local_floor},
            {"local_cap", local_cap},
            {"global_floor", global_floor},
            {
                "global_cap",
                global_cap(
                    generator,
                    global_floor,
                    maximum_aggregate,
                    0.25f,
                    0.85f,
                    0.75f
                )
            },
        });
    }

    for (std::size_t row = 0U; row < tail_row_count; ++row) {
        const auto [maturity, interval, observation_count] = schedule(
            generator, kTailIntervals, 126U, 1764U
        );
        const float local_floor = uniform(generator, -0.25f, 0.0f);
        const float local_cap = uniform(generator, 0.005f, 0.30f);
        const float minimum_aggregate =
            static_cast<float>(observation_count) * local_floor;
        const float maximum_aggregate =
            static_cast<float>(observation_count) * local_cap;
        const float global_floor = uniform(
            generator, std::max(-0.50f, 0.80f * minimum_aggregate), 0.0f
        );
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {"participation_rate", uniform(generator, 0.25f, 2.0f)},
            {"local_floor", local_floor},
            {"local_cap", local_cap},
            {"global_floor", global_floor},
            {
                "global_cap",
                global_cap(
                    generator,
                    global_floor,
                    maximum_aggregate,
                    0.10f,
                    0.95f,
                    1.50f
                )
            },
        });
    }
    generated.construction = {
        {"method", "ordered 90/10 constrained core-stress sample"},
        {"row_order", {
            {"core", "rows 1-" + std::to_string(core_row_count)},
            {"stress", "remaining rows"},
        }},
        {"core_share", 0.9},
        {"stress_share", 0.1},
        {"valuation_time", "issuance"},
        {"reference_spot", 1.0},
        {"nominal", 1.0},
        {"sampling", {
            {
                "regime_counts",
                std::to_string(core_row_count) + " core rows and "
                    + std::to_string(tail_row_count) + " stress rows"
            },
            {
                "observation_interval",
                "uniform categorical draw within the selected regime"
            },
            {
                "observation_count",
                "uniform integer draw compatible with maturity bounds"
            },
            {
                "continuous_parameters",
                "uniform draws within the selected regime bounds"
            },
            {
                "global_bounds",
                "conditional draws from the theoretical sum of local bounds"
            },
        }},
        {"calendar_rules", {
            "maturity is an integer multiple of observation_interval",
            "observation dates are i * observation_interval from issuance",
            "the first return compares S at issuance with the first observation",
            "core rows precede stress rows",
        }},
        {"regimes", {
            {"core", {
                {"row_count", core_row_count},
                {"maturity", {252, 1260}},
                {"observation_interval", {21, 63}},
                {"participation_rate", {0.80, 1.20}},
                {"local_floor", {-0.05, 0.0}},
                {"local_cap", {0.01, 0.08}},
                {"global_floor", {0.0, 0.03}},
                {
                    "global_cap",
                    "25% to 85% of the bounded theoretical local-cap sum"
                },
            }},
            {"stress", {
                {"row_count", tail_row_count},
                {"maturity", {126, 1764}},
                {"observation_interval", {5, 126, 252}},
                {"participation_rate", {0.25, 2.0}},
                {"local_floor", {-0.25, 0.0}},
                {"local_cap", {0.005, 0.30}},
                {"global_floor", {-0.50, 0.0}},
                {
                    "global_cap",
                    "10% to 95% of the bounded theoretical local-cap sum"
                },
            }},
        }},
        {"constraints", {
            "local_floor < local_cap",
            "-1 < global_floor < global_cap",
                    "global_cap is bounded by 0.75 in core and 1.50 in stress",
        }},
    };
    return generated;
}

}  // namespace ai_factory::workbench::datasets::cliquet
