// Host implementation of constrained core-tail autocall generation.
#include "tools/datasets/autocall_generation.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <utility>

namespace ai_factory::workbench::datasets::autocall {
namespace {

constexpr std::array<float, 2U> kCoreIntervals = {
    1.0f / 12.0f, 1.0f / 4.0f,
};
constexpr std::array<float, 3U> kTailIntervals = {
    1.0f / 52.0f, 1.0f / 2.0f, 1.0f,
};

// Draw one uniform FP32 value without exposing RNG details to generators.
float uniform(
    std::mt19937_64& generator,
    float minimum,
    float maximum
) {
    return std::uniform_real_distribution<float>(minimum, maximum)(generator);
}

// Draw an issue schedule whose maturity is an exact interval multiple.
template <std::size_t IntervalCount>
std::pair<float, float> schedule(
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
    };
}

// Describe common issuance schedules and the exact 90/10 sampling policy.
nlohmann::ordered_json common_construction(
    std::size_t core_row_count,
    std::size_t tail_row_count
) {
    return {
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
                "uniform integer draw among counts compatible with the "
                "regime maturity bounds"
            },
            {
                "continuous_parameters",
                "uniform draws within the selected regime bounds"
            },
        }},
        {"calendar_rules", {
            "maturity is an integer multiple of observation_interval",
            "observation dates are i * observation_interval from issuance",
            "core rows precede stress rows",
        }},
    };
}

// Reject empty regimes before constructing their distributions.
void validate_counts(
    std::size_t core_row_count,
    std::size_t tail_row_count
) {
    if (core_row_count == 0U || tail_row_count == 0U) {
        throw std::invalid_argument(
            "Autocall generation requires non-empty core and stress regimes."
        );
    }
}

}  // namespace

// Generate coupon-barrier terms shared by Phoenix and Phoenix Memory.
GeneratedRows generate_phoenix_rows(
    std::size_t core_row_count,
    std::size_t tail_row_count,
    std::uint64_t seed
) {
    validate_counts(core_row_count, tail_row_count);
    std::mt19937_64 generator(seed);
    GeneratedRows generated;
    generated.rows.reserve(core_row_count + tail_row_count);

    for (std::size_t row = 0U; row < core_row_count; ++row) {
        const auto [maturity, interval] = schedule(
            generator, kCoreIntervals, 1.0f, 5.0f
        );
        const float protection_barrier = uniform(generator, 0.50f, 0.70f);
        const float coupon_barrier = uniform(
            generator, std::max(protection_barrier, 0.55f), 0.85f
        );
        const float autocall_barrier = uniform(
            generator, std::max(coupon_barrier, 0.95f), 1.10f
        );
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {"autocall_barrier", autocall_barrier},
            {"coupon_barrier", coupon_barrier},
            {"protection_barrier", protection_barrier},
            {"annual_coupon_rate", uniform(generator, 0.03f, 0.15f)},
        });
    }

    for (std::size_t row = 0U; row < tail_row_count; ++row) {
        const auto [maturity, interval] = schedule(
            generator, kTailIntervals, 0.5f, 7.0f
        );
        const float protection_barrier = uniform(generator, 0.20f, 0.90f);
        const float coupon_barrier = uniform(
            generator, std::max(protection_barrier, 0.30f), 1.05f
        );
        const float autocall_barrier = uniform(
            generator, std::max(coupon_barrier, 0.80f), 1.30f
        );
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {"autocall_barrier", autocall_barrier},
            {"coupon_barrier", coupon_barrier},
            {"protection_barrier", protection_barrier},
            {"annual_coupon_rate", uniform(generator, 0.005f, 0.30f)},
        });
    }
    generated.construction = common_construction(
        core_row_count, tail_row_count
    );
    generated.construction["regimes"] = {
        {"core", {
            {"row_count", core_row_count},
            {"maturity", {"1 year", "5 years"}},
            {"observation_interval", {"1 / 12", "1 / 4"}},
            {"autocall_barrier", {0.95, 1.10}},
            {"coupon_barrier", {0.55, 0.85}},
            {"protection_barrier", {0.50, 0.70}},
            {"annual_coupon_rate", {0.03, 0.15}},
        }},
        {"stress", {
            {"row_count", tail_row_count},
            {"maturity", {"0.5 year", "7 years"}},
            {"observation_interval", {"1 / 52", "1 / 2", "1"}},
            {"autocall_barrier", {0.80, 1.30}},
            {"coupon_barrier", {0.30, 1.05}},
            {"protection_barrier", {0.20, 0.90}},
            {"annual_coupon_rate", {0.005, 0.30}},
        }},
    };
    generated.construction["sampling"]["barriers"] =
        "draw protection first, then coupon above protection, then "
        "autocall above coupon";
    generated.construction["barrier_rule"] =
        "protection_barrier <= coupon_barrier <= autocall_barrier";
    return generated;
}

// Generate accumulated-gain Athena terms without a coupon barrier.
GeneratedRows generate_athena_rows(
    std::size_t core_row_count,
    std::size_t tail_row_count,
    std::uint64_t seed
) {
    validate_counts(core_row_count, tail_row_count);
    std::mt19937_64 generator(seed);
    GeneratedRows generated;
    generated.rows.reserve(core_row_count + tail_row_count);

    for (std::size_t row = 0U; row < core_row_count; ++row) {
        const auto [maturity, interval] = schedule(
            generator, kCoreIntervals, 1.0f, 5.0f
        );
        const float protection_barrier = uniform(generator, 0.50f, 0.70f);
        const float autocall_barrier = uniform(
            generator, std::max(protection_barrier, 0.95f), 1.10f
        );
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {"autocall_barrier", autocall_barrier},
            {"protection_barrier", protection_barrier},
            {"annual_coupon_rate", uniform(generator, 0.03f, 0.15f)},
        });
    }

    for (std::size_t row = 0U; row < tail_row_count; ++row) {
        const auto [maturity, interval] = schedule(
            generator, kTailIntervals, 0.5f, 7.0f
        );
        const float protection_barrier = uniform(generator, 0.20f, 0.90f);
        const float autocall_barrier = uniform(
            generator, std::max(protection_barrier, 0.80f), 1.30f
        );
        generated.rows.push_back({
            {"maturity", maturity},
            {"observation_interval", interval},
            {"autocall_barrier", autocall_barrier},
            {"protection_barrier", protection_barrier},
            {"annual_coupon_rate", uniform(generator, 0.005f, 0.30f)},
        });
    }
    generated.construction = common_construction(
        core_row_count, tail_row_count
    );
    generated.construction["regimes"] = {
        {"core", {
            {"row_count", core_row_count},
            {"maturity", {"1 year", "5 years"}},
            {"observation_interval", {"1 / 12", "1 / 4"}},
            {"autocall_barrier", {0.95, 1.10}},
            {"protection_barrier", {0.50, 0.70}},
            {"annual_coupon_rate", {0.03, 0.15}},
        }},
        {"stress", {
            {"row_count", tail_row_count},
            {"maturity", {"0.5 year", "7 years"}},
            {"observation_interval", {"1 / 52", "1 / 2", "1"}},
            {"autocall_barrier", {0.80, 1.30}},
            {"protection_barrier", {0.20, 0.90}},
            {"annual_coupon_rate", {0.005, 0.30}},
        }},
    };
    generated.construction["sampling"]["barriers"] =
        "draw protection first, then autocall above protection";
    generated.construction["barrier_rule"] =
        "protection_barrier <= autocall_barrier";
    return generated;
}

}  // namespace ai_factory::workbench::datasets::autocall
