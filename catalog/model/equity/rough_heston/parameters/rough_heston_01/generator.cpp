// Generate ordered 90/10 rough-Heston parameter rows.
#include "model/equity/rough/rough_heston/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::datasets;

    constexpr std::uint64_t seed = 710001101ULL;
    GeneratedRows core = uniform_rows(900U, seed, {
        {"spot", 1.0f, 1.0f},
        {"risk_free_rate", 0.001f, 0.08f},
        {"dividend_yield", 0.0f, 0.06f},
        {"initial_variance", 0.01f, 0.12f},
        {"mean_reversion", 0.5f, 4.0f},
        {"long_run_variance", 0.01f, 0.15f},
        {"hurst_exponent", 0.03f, 0.25f},
        {"rho", -0.95f, -0.25f},
    });
    GeneratedRows stress = uniform_rows(100U, seed + 2U, {
        {"spot", 1.0f, 1.0f},
        {"risk_free_rate", -0.03f, 0.12f},
        {"dividend_yield", 0.0f, 0.10f},
        {"initial_variance", 0.003f, 0.30f},
        {"mean_reversion", 0.10f, 8.0f},
        {"long_run_variance", 0.003f, 0.35f},
        {"hurst_exponent", 0.01f, 0.45f},
        {"rho", -0.99f, 0.25f},
    });

    const auto reconstruct_coefficients = [](
        GeneratedRows& regime,
        std::uint64_t volatility_seed,
        float denominator,
        float numerator,
        float absolute_minimum,
        float absolute_maximum
    ) {
        std::mt19937_64 generator(volatility_seed);
        for (ParameterRow& row : regime.rows) {
            const float mean_reversion =
                row.at("mean_reversion").get<float>();
            const float long_run_variance =
                row.at("long_run_variance").get<float>();
            const float variance_drift =
                mean_reversion * long_run_variance;
            const float minimum = std::max(
                std::sqrt(variance_drift / denominator), absolute_minimum
            );
            const float maximum = std::min(
                std::sqrt(numerator * variance_drift), absolute_maximum
            );
            const float volatility_of_variance =
                std::uniform_real_distribution<float>(minimum, maximum)(
                    generator
                );
            row = {
                {"spot", row.at("spot")},
                {"risk_free_rate", row.at("risk_free_rate")},
                {"dividend_yield", row.at("dividend_yield")},
                {"initial_variance", row.at("initial_variance")},
                {"mean_reversion", mean_reversion},
                {"variance_drift", variance_drift},
                {"volatility_of_variance", volatility_of_variance},
                {"hurst_exponent", row.at("hurst_exponent")},
                {"rho", row.at("rho")},
            };
        }
        regime.construction["conditional_reconstruction"] = {
            {
                "variance_drift",
                "mean_reversion * sampled long_run_variance"
            },
            {"implied_long_run_variance", "variance_drift / mean_reversion"},
            {"volatility_of_variance", {
                {"minimum", "max(sqrt(variance_drift / "
                    + std::to_string(denominator) + "), "
                    + std::to_string(absolute_minimum) + ")"},
                {"maximum", "min(sqrt("
                    + std::to_string(numerator)
                    + " * variance_drift), "
                    + std::to_string(absolute_maximum) + ")"},
                {
                    "distribution",
                    "uniform conditional on reconstructed variance_drift"
                },
            }},
        };
    };
    reconstruct_coefficients(
        core, seed + 1U, 5.0f, 12.0f, 0.08f, 0.8f
    );
    reconstruct_coefficients(
        stress, seed + 3U, 8.0f, 20.0f, 0.03f, 1.8f
    );
    GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );
    rows.construction["variance_parameterization"] = {
        {
            "equation",
            "theta is variance_drift in theta - mean_reversion * V_t"
        },
        {
            "reason",
            "sample a readable long-run variance, then store production coefficients"
        },
    };
    rows.construction["rough_kernel"] = {
        {"formula", "K_H(u) = u^(H-1/2) / Gamma(H+1/2)"},
        {"production_scheme", "positive exponential N-factor Markovian lift"},
    };

    const std::filesystem::path dataset =
        "datasets/model/equity/rough_heston/parameters/rough_heston_01.json";
    write_model_dataset(
        "rough_heston_01",
        "rough Heston",
        dataset,
        "catalog/model/equity/rough_heston/parameters/rough_heston_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/"
        "rough_heston/parameters/rough_heston_01.json",
        {
            {"spot", "Initial spot; fixed to 1 in this dataset."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_variance", "Initial instantaneous variance V0."},
            {"mean_reversion", "Linear variance mean-reversion coefficient."},
            {
                "variance_drift",
                "Constant variance drift theta; implied long-run variance is "
                "theta/lambda."
            },
            {"volatility_of_variance", "Variance diffusion coefficient nu."},
            {
                "hurst_exponent",
                "Roughness exponent H, strictly between 0 and 0.5."
            },
            {"rho", "Spot/variance-driver Brownian correlation."},
        },
        {
            {"spot", "dS_t / S_t = (r-q) dt + sqrt(V_t) dZ_t."},
            {
                "variance",
                "V_t = V0 + integral_0^t K_H(t-s) "
                "[(theta-lambda V_s) ds + nu sqrt(V_s) dW_s]."
            },
            {
                "kernel",
                "K_H(u) = u^(H-1/2) / Gamma(H+1/2)."
            },
            {"correlation", "d<Z,W>_t = rho dt."},
            {
                "simulation",
                "Positive exponential N-factor approximation of the Volterra kernel."
            },
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::rough_heston::load_models(dataset).size() != 1000U) {
        throw std::runtime_error(
            "The generated rough-Heston dataset did not reload 1000 rows."
        );
    }
}
