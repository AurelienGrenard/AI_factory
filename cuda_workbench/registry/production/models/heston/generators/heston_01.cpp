// Generate Heston rows by sampling ordinary parameters first, then gamma from
// row-dependent bounds that control the range of the Feller ratio.
#include "tools/registry/parameter_database.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>

int main() {
    using namespace ai_factory::workbench::registry;
    using nlohmann::ordered_json;

    const std::filesystem::path json_path =
        "cuda_workbench/registry/production/models/heston/data/heston_01.json";
    const std::filesystem::path generator_path =
        "cuda_workbench/registry/production/models/heston/generators/heston_01.cpp";

    constexpr std::uint64_t seed = 710000201ULL;
    constexpr std::uint64_t gamma_seed = seed + 1ULL;

    // First sample every parameter that has fixed, row-independent bounds.
    GeneratedRows rows = uniform_rows(1'000U, seed, {
        {"spot", 1.0f, 1.0f},
        {"risk_free_rate", 0.0f, 0.08f},
        {"dividend_yield", 0.0f, 0.05f},
        {"initial_variance", 0.01f, 0.12f},
        {"kappa", 0.5f, 4.0f},
        {"theta", 0.01f, 0.15f},
        {"rho", -1.0f, -0.30f},
    });

    // Gamma is conditional on the already sampled kappa and theta of each row.
    std::mt19937_64 gamma_generator(gamma_seed);
    for (ParameterRow& row : rows.rows) {
        const float kappa = row.at("kappa").get<float>();
        const float theta = row.at("theta").get<float>();
        const float gamma_min = std::max(std::sqrt(kappa * theta / 5.0f), 0.1f);
        const float gamma_max = std::min(std::sqrt(12.0f * kappa * theta), 0.8f);
        std::uniform_real_distribution<float> gamma_distribution(
            gamma_min, gamma_max
        );
        row["gamma"] = gamma_distribution(gamma_generator);
    }

    rows.construction["method"] = "conditional uniform sample";
    rows.construction["conditional_bounds"] = {
        {"gamma", {
            {"minimum", "max(sqrt(kappa * theta / 5), 0.1)"},
            {"maximum", "min(sqrt(12 * kappa * theta), 0.8)"},
        }},
    };
    rows.construction["feller_ratio"] = {
        {"expression", "2 * kappa * theta / gamma^2"},
        {"controlled_range", {1.0 / 6.0, 10.0}},
    };

    write_model_database(
        "heston_01",
        "Heston",
        json_path,
        generator_path,
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_variance", "Initial variance v0."},
            {"kappa", "Variance mean-reversion speed."},
            {"theta", "Long-run variance."},
            {"gamma", "Volatility of variance."},
            {"rho", "Spot/variance Brownian correlation."},
        },
        {
            {"spot", "dS_t / S_t = (r - q) dt + sqrt(V_t) dW_t^S"},
            {"variance", "dV_t = kappa (theta - V_t) dt + gamma sqrt(V_t) dW_t^V"},
            {"correlation", "d<W^S, W^V>_t = rho dt"},
        },
        rows
    );
}
