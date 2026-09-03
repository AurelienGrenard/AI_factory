// Generate ordinary Bates rows first and place a smaller stress regime at the
// end of the dataset so atypical jump and variance combinations stay visible.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <string>
#include <utility>

namespace {

using ai_factory::workbench::datasets::GeneratedRows;
using ai_factory::workbench::datasets::ParameterRow;

// Sample gamma conditionally so the variance process spans useful Feller
// ratios without combining tiny mean levels with disproportionate vol-of-vol.
void assign_conditional_gamma(
    GeneratedRows& rows,
    std::uint64_t seed,
    float feller_denominator,
    float feller_numerator,
    float absolute_minimum,
    float absolute_maximum
) {
    std::mt19937_64 generator(seed);
    for (ParameterRow& row : rows.rows) {
        const float kappa = row.at("kappa").get<float>();
        const float theta = row.at("theta").get<float>();
        const float minimum = std::max(
            std::sqrt(kappa * theta / feller_denominator),
            absolute_minimum
        );
        const float maximum = std::min(
            std::sqrt(feller_numerator * kappa * theta),
            absolute_maximum
        );
        std::uniform_real_distribution<float> distribution(minimum, maximum);
        row["gamma"] = distribution(generator);
    }
}

}  // namespace

// Generate the Bates dataset and its concise, self-contained catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/equity/bates/parameters/bates_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/equity/bates/parameters/bates_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/equity/bates/parameters/bates_01.json";

    constexpr std::uint64_t seed = 710000301ULL;

    GeneratedRows ordinary = uniform_rows(900U, seed, {
        {"spot", 1.0f, 1.0f},
        {"risk_free_rate", 0.001f, 0.08f},
        {"dividend_yield", 0.0f, 0.06f},
        {"initial_variance", 0.01f, 0.12f},
        {"kappa", 0.5f, 4.0f},
        {"theta", 0.01f, 0.15f},
        {"rho", -0.95f, -0.25f},
        {"jump_intensity", 0.02f, 1.0f},
        {"jump_log_mean", -0.25f, 0.05f},
        {"jump_log_volatility", 0.05f, 0.35f},
    });
    assign_conditional_gamma(
        ordinary, seed + 1ULL, 5.0f, 12.0f, 0.1f, 0.8f
    );

    GeneratedRows stressed = uniform_rows(100U, seed + 2ULL, {
        {"spot", 1.0f, 1.0f},
        {"risk_free_rate", -0.03f, 0.12f},
        {"dividend_yield", 0.0f, 0.10f},
        {"initial_variance", 0.005f, 0.25f},
        {"kappa", 0.15f, 6.0f},
        {"theta", 0.005f, 0.30f},
        {"rho", -0.99f, 0.10f},
        {"jump_intensity", 1.0f, 2.5f},
        {"jump_log_mean", -0.45f, 0.05f},
        {"jump_log_volatility", 0.30f, 0.55f},
    });
    assign_conditional_gamma(
        stressed, seed + 3ULL, 8.0f, 20.0f, 0.05f, 1.5f
    );

    ordinary.construction["conditional_sampling"] = {
        {"gamma", {
            {"minimum", "max(sqrt(kappa * theta / 5), 0.1)"},
            {"maximum", "min(sqrt(12 * kappa * theta), 0.8)"},
            {"distribution", "uniform conditional on kappa and theta"},
        }},
    };
    stressed.construction["conditional_sampling"] = {
        {"gamma", {
            {"minimum", "max(sqrt(kappa * theta / 8), 0.05)"},
            {"maximum", "min(sqrt(20 * kappa * theta), 1.5)"},
            {"distribution", "uniform conditional on kappa and theta"},
        }},
    };
    GeneratedRows rows = core_stress_rows(
        std::move(ordinary), std::move(stressed)
    );
    rows.construction["jump_convention"] = {
            {"law", "log(Y) ~ Normal(jump_log_mean, jump_log_volatility^2)"},
            {"count", "N(dt) ~ Poisson(jump_intensity * dt)"},
            {"martingale_compensator", "jump_intensity * (E[Y] - 1)"},
    };

    write_model_dataset(
        "bates_01",
        "Bates",
        dataset_path,
        catalog_path,
        url,
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_variance", "Initial variance v0."},
            {"kappa", "Variance mean-reversion speed."},
            {"theta", "Long-run variance."},
            {"gamma", "Volatility of variance."},
            {"rho", "Spot/variance Brownian correlation."},
            {"jump_intensity", "Poisson jump intensity lambda per year."},
            {"jump_log_mean", "Mean nu of the log jump size."},
            {"jump_log_volatility", "Standard deviation delta of the log jump size."},
        },
        {
            {"spot", "dS_t / S_(t-) = (r - q - lambda * k_J) dt + sqrt(V_t) dW_t^S + (Y - 1) dN_t"},
            {"variance", "dV_t = kappa (theta - V_t) dt + gamma sqrt(V_t) dW_t^V"},
            {"correlation", "d<W^S, W^V>_t = rho dt"},
            {"jump_size", "log(Y) ~ Normal(nu, delta^2)"},
            {"compensator", "k_J = exp(nu + delta^2 / 2) - 1"},
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
