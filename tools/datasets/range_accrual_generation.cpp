// Host implementation of constrained core-tail Range Accrual generation.
#include "tools/datasets/range_accrual_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace ai_factory::workbench::datasets::range_accrual {
namespace {

constexpr std::array<float, 3U> kCoreIntervals = {
    1.0f / 252.0f, 1.0f / 52.0f, 1.0f / 12.0f,
};
constexpr std::array<float, 4U> kTailIntervals = {
    1.0f / 252.0f, 1.0f / 52.0f, 1.0f / 12.0f, 1.0f / 4.0f,
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
std::tuple<float, float, std::size_t> schedule(
    std::mt19937_64& generator,
    const std::array<float, IntervalCount>& intervals,
    float minimum_maturity,
    float maximum_maturity
) {
    const float interval = intervals[
        std::uniform_int_distribution<std::size_t>(
            0U, intervals.size() - 1U
        )(generator)
    ];
    const auto minimum_observations = std::max<std::size_t>(
        1U,
        static_cast<std::size_t>(ceilf(minimum_maturity / interval))
    );
    const auto maximum_observations = static_cast<std::size_t>(
        floorf(maximum_maturity / interval)
    );
    const std::size_t observation_count =
        std::uniform_int_distribution<std::size_t>(
            minimum_observations, maximum_observations
        )(generator);
    return {
        interval * static_cast<float>(observation_count),
        interval,
        observation_count,
    };
}

// Reject empty regimes before constructing their distributions.
void validate_counts(
    std::size_t core_row_count,
    std::size_t tail_row_count
) {
    if (core_row_count == 0U || tail_row_count == 0U) {
        throw std::invalid_argument(
            "Range Accrual generation requires non-empty core and tail regimes."
        );
    }
}

}  // namespace

// Generate valid observation bands and shuffle both regimes.
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
            generator, kCoreIntervals, 1.0f, 5.0f
        );
        static_cast<void>(observation_count);
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {"lower_barrier", uniform(generator, 0.70f, 0.95f)},
            {"upper_barrier", uniform(generator, 1.05f, 1.40f)},
            {"coupon_rate", uniform(generator, 0.02f, 0.12f)},
        });
    }

    for (std::size_t row = 0U; row < tail_row_count; ++row) {
        const auto [maturity, interval, observation_count] = schedule(
            generator, kTailIntervals, 0.5f, 7.0f
        );
        static_cast<void>(observation_count);
        const bool narrow_range = (row % 2U) == 0U;
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {
                "lower_barrier",
                narrow_range
                    ? uniform(generator, 0.94f, 0.995f)
                    : uniform(generator, 0.25f, 0.70f)
            },
            {
                "upper_barrier",
                narrow_range
                    ? uniform(generator, 1.005f, 1.06f)
                    : uniform(generator, 1.40f, 2.50f)
            },
            {"coupon_rate", uniform(generator, 0.005f, 0.25f)},
        });
    }
    std::shuffle(generated.rows.begin(), generated.rows.end(), generator);

    generated.construction = {
        {"method", "constrained core-tail random sample"},
        {"valuation_time", "issuance"},
        {"reference_spot", 1.0},
        {"nominal", 1.0},
        {"sampling", {
            {
                "regime_counts",
                std::to_string(core_row_count) + " core rows and "
                    + std::to_string(tail_row_count) + " tail rows"
            },
            {
                "observation_interval",
                "uniform categorical draw within the selected regime"
            },
            {
                "observation_count",
                "uniform integer draw compatible with maturity bounds"
            },
            {"continuous_parameters", "uniform draws within each regime"},
        }},
        {"calendar_rules", {
            "maturity is an integer multiple of observation_interval",
            "observation dates are i * observation_interval from issuance",
            "the issuance spot is normalized to one and is not observed",
            "maturity is included as the final observation date",
            "rows are shuffled after both regimes are generated",
        }},
        {"regimes", {
            {"core", {
                {"row_count", core_row_count},
                {"maturity", {"1 year", "5 years"}},
                {"observation_interval", {"1 / 252", "1 / 52", "1 / 12"}},
                {"lower_barrier", {0.70, 0.95}},
                {"upper_barrier", {1.05, 1.40}},
                {"coupon_rate", {0.02, 0.12}},
            }},
            {"tail", {
                {"row_count", tail_row_count},
                {"maturity", {"0.5 year", "7 years"}},
                {"observation_interval", {
                    "1 / 252", "1 / 52", "1 / 12", "1 / 4"
                }},
                {"range_shape", "50 narrow rows and 50 wide rows"},
                {"narrow_lower_barrier", {0.94, 0.995}},
                {"narrow_upper_barrier", {1.005, 1.06}},
                {"wide_lower_barrier", {0.25, 0.70}},
                {"wide_upper_barrier", {1.40, 2.50}},
                {"coupon_rate", {0.005, 0.25}},
            }},
        }},
        {"constraints", {
            "0 < lower_barrier < 1 < upper_barrier",
            "coupon_rate is positive",
        }},
    };
    return generated;
}

}  // namespace ai_factory::workbench::datasets::range_accrual
