// Generate ordered 90/10 Merton jump-diffusion parameter rows.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>

int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint64_t seed = 710000601ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"volatility", 0.08f, 0.45f},
            {"jump_intensity", 0.02f, 1.0f},
            {"jump_log_mean", -0.20f, 0.08f},
            {"jump_log_volatility", 0.03f, 0.30f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"volatility", 0.03f, 0.80f},
            {"jump_intensity", 0.001f, 3.0f},
            {"jump_log_mean", -0.55f, 0.30f},
            {"jump_log_volatility", 0.005f, 0.70f},
        })
    );
    for (auto& row : rows.rows) {
        ParameterRow with_spot = {{"spot", 1.0f}};
        for (const auto& [name, value] : row.items()) {
            with_spot[name] = value;
        }
        row = std::move(with_spot);
    }
    const std::filesystem::path dataset =
        "datasets/model/equity/merton/merton_01.json";
    write_model_dataset(
        "merton_01",
        "Merton jump diffusion",
        dataset,
        "catalog/model/equity/merton/merton_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/merton/merton_01.json",
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"volatility", "Diffusion volatility."},
            {"jump_intensity", "Poisson jump intensity."},
            {"jump_log_mean", "Mean log jump size."},
            {"jump_log_volatility", "Log jump-size volatility."},
        },
        {
            {
                "log_spot",
                "Brownian diffusion plus compound-Poisson normal jumps."
            },
            {
                "martingale_correction",
                "lambda * (exp(mu_J + delta_J^2/2) - 1)."
            },
            {
                "simulation",
                "Exact independent increments; normal jumps aggregated "
                "conditionally on their Poisson count."
            },
        },
        rows
    );
    validate_model_dataset_file(dataset);
}
