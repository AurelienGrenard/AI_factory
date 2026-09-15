// Conditional volatility draws and complete recipe metadata for CIR-factor rows.
#include "tools/sampling/parameters/cir_generation.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using ai_factory::workbench::datasets::GeneratedRows;
using ai_factory::workbench::datasets::ParameterRow;

// Draw sigma conditionally and record the complete reproducible construction.
void assign_volatility(
    GeneratedRows& regime,
    std::uint64_t volatility_seed,
    float denominator,
    float numerator,
    float absolute_minimum,
    float absolute_maximum,
    const std::string& feller_minimum,
    float feller_maximum
) {
    std::mt19937_64 generator(volatility_seed);
    for (ParameterRow& row : regime.rows) {
        const float product = row.at("mean_reversion").get<float>()
            * row.at("long_term_mean").get<float>();
        const float minimum = std::max(
            std::sqrt(product / denominator), absolute_minimum
        );
        const float maximum = std::min(
            std::sqrt(numerator * product), absolute_maximum
        );
        if (!(minimum <= maximum)) {
            throw std::logic_error(
                "CIR volatility bounds are empty for one generated row."
            );
        }
        const float volatility = std::uniform_real_distribution<float>(
            minimum, maximum
        )(generator);
        const float feller_ratio = 2.0f * product
            / (volatility * volatility);
        const float theoretical_minimum = 2.0f / numerator;
        if (!(feller_ratio >= theoretical_minimum - 1.0e-5f
              && feller_ratio <= feller_maximum + 1.0e-4f)) {
            throw std::logic_error(
                "Generated CIR row violates its Feller-ratio bounds."
            );
        }
        row["volatility"] = volatility;
    }
    regime.construction["conditional_sampling"] = {
        {"volatility", {
            {
                "minimum",
                "max(sqrt(mean_reversion * long_term_mean / "
                    + std::to_string(denominator) + "), "
                    + std::to_string(absolute_minimum) + ")"
            },
            {
                "maximum",
                "min(sqrt(" + std::to_string(numerator)
                    + " * mean_reversion * long_term_mean), "
                    + std::to_string(absolute_maximum) + ")"
            },
            {
                "distribution",
                "uniform conditional on mean_reversion and long_term_mean"
            },
            {"seed", volatility_seed},
        }},
        {"feller_ratio", {
            {
                "definition",
                "2 * mean_reversion * long_term_mean / volatility^2"
            },
            {"guaranteed_minimum", feller_minimum},
            {"guaranteed_maximum", feller_maximum},
            {"feller_condition_threshold", 1.0},
        }},
        {
            "purpose",
            "cover accessible and inaccessible zero-boundary regimes"
        },
    };
}

}  // namespace

namespace ai_factory::workbench::datasets::cir {

GeneratedRows generate_core_stress_rows(std::uint64_t seed) {
    GeneratedRows core = uniform_rows(900U, seed, {
        {"mean_reversion", 0.03f, 1.0f},
        {"long_term_mean", 0.001f, 0.08f},
        {"initial_state", 0.001f, 0.08f},
    });
    GeneratedRows stress = uniform_rows(100U, seed + 2ULL, {
        {"mean_reversion", 0.005f, 2.5f},
        {"long_term_mean", 0.0001f, 0.20f},
        {"initial_state", 0.0f, 0.20f},
    });
    core.construction["seed"] = seed;
    stress.construction["seed"] = seed + 2ULL;

    assign_volatility(
        core,
        seed + 1ULL,
        5.0f,
        12.0f,
        0.005f,
        0.30f,
        "1 / 6",
        10.0f
    );
    assign_volatility(
        stress,
        seed + 3ULL,
        8.0f,
        20.0f,
        0.001f,
        0.80f,
        "1 / 10",
        16.0f
    );
    return core_stress_rows(std::move(core), std::move(stress));
}

}  // namespace ai_factory::workbench::datasets::cir
