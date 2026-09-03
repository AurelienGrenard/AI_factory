// Generate Heston rows by sampling ordinary parameters first, then gamma from
// row-dependent bounds that control the range of the Feller ratio.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <random>
#include <string>
#include <utility>

// Generate the Heston dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;
    using nlohmann::ordered_json;

    const std::filesystem::path dataset_path =
        "datasets/model/equity/heston/parameters/heston_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/equity/heston/parameters/heston_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/equity/heston/parameters/heston_01.json";

    constexpr std::uint64_t seed = 710000201ULL;
    GeneratedRows core = uniform_rows(900U, seed, {
        {"spot", 1.0f, 1.0f},
        {"risk_free_rate", 0.001f, 0.08f},
        {"dividend_yield", 0.0f, 0.06f},
        {"initial_variance", 0.01f, 0.12f},
        {"kappa", 0.5f, 4.0f},
        {"theta", 0.01f, 0.15f},
        {"rho", -0.95f, -0.25f},
    });
    GeneratedRows stress = uniform_rows(100U, seed + 2ULL, {
        {"spot", 1.0f, 1.0f},
        {"risk_free_rate", -0.03f, 0.12f},
        {"dividend_yield", 0.0f, 0.10f},
        {"initial_variance", 0.003f, 0.30f},
        {"kappa", 0.10f, 8.0f},
        {"theta", 0.003f, 0.35f},
        {"rho", -0.99f, 0.25f},
    });

    const auto assign_gamma = [](GeneratedRows& regime,
                                 std::uint64_t gamma_seed,
                                 float denominator,
                                 float numerator,
                                 float absolute_minimum,
                                 float absolute_maximum) {
        std::mt19937_64 generator(gamma_seed);
        for (ParameterRow& row : regime.rows) {
            const float product = row.at("kappa").get<float>()
                * row.at("theta").get<float>();
            const float minimum = std::max(
                std::sqrt(product / denominator), absolute_minimum
            );
            const float maximum = std::min(
                std::sqrt(numerator * product), absolute_maximum
            );
            row["gamma"] = std::uniform_real_distribution<float>(
                minimum, maximum
            )(generator);
        }
        regime.construction["conditional_sampling"] = {
            {"gamma", {
                {"minimum", "max(sqrt(kappa * theta / "
                    + std::to_string(denominator) + "), "
                    + std::to_string(absolute_minimum) + ")"},
                {"maximum", "min(sqrt("
                    + std::to_string(numerator)
                    + " * kappa * theta), "
                    + std::to_string(absolute_maximum) + ")"},
                {"distribution", "uniform conditional on kappa and theta"},
            }},
            {"purpose", "control the Feller ratio while keeping gamma plausible"},
        };
    };
    assign_gamma(core, seed + 1ULL, 5.0f, 12.0f, 0.08f, 0.8f);
    assign_gamma(stress, seed + 3ULL, 8.0f, 20.0f, 0.03f, 1.8f);
    GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_model_dataset(
        "heston_01",
        "Heston",
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
        },
        {
            {"spot", "dS_t / S_t = (r - q) dt + sqrt(V_t) dW_t^S"},
            {"variance", "dV_t = kappa (theta - V_t) dt + gamma sqrt(V_t) dW_t^V"},
            {"correlation", "d<W^S, W^V>_t = rho dt"},
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
