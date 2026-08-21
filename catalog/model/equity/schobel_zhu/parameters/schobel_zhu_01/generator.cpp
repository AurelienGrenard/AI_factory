// Generate ordered 90/10 Schobel-Zhu stochastic-volatility parameter rows.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>

int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint64_t seed = 710000901ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"initial_volatility", 0.08f, 0.45f},
            {"mean_reversion", 0.30f, 5.0f},
            {"long_run_volatility", 0.08f, 0.45f},
            {"volatility_of_volatility", 0.03f, 0.60f},
            {"correlation", -0.90f, 0.30f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"initial_volatility", 0.01f, 0.80f},
            {"mean_reversion", 0.03f, 10.0f},
            {"long_run_volatility", 0.01f, 0.80f},
            {"volatility_of_volatility", 0.005f, 1.30f},
            {"correlation", -0.98f, 0.90f},
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
        "datasets/model/equity/schobel_zhu/parameters/schobel_zhu_01.json";
    write_model_dataset(
        "schobel_zhu_01",
        "Schobel-Zhu",
        dataset,
        "catalog/model/equity/schobel_zhu/parameters/schobel_zhu_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/schobel_zhu/parameters/schobel_zhu_01.json",
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {
                "initial_volatility",
                "Initial signed OU volatility factor."
            },
            {"mean_reversion", "Positive OU mean-reversion speed."},
            {"long_run_volatility", "Long-run signed OU level."},
            {"volatility_of_volatility", "OU diffusion scale."},
            {"correlation", "Instantaneous Brownian correlation."},
        },
        {
            {"spot", "dS/S = (r-q)dt + v dW_S."},
            {
                "volatility",
                "dv = kappa(theta-v)dt + xi dW_v."
            },
            {
                "simulation",
                "Exact OU endpoint coupled consistently to a log-spot "
                "Euler step."
            },
        },
        rows
    );
    validate_model_dataset_file(dataset);
}
