// Generate ordered 90/10 CEV local-volatility parameter rows.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>

int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint64_t seed = 710000801ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"sigma", 0.08f, 0.45f},
            {"beta", 0.55f, 0.95f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"sigma", 0.03f, 0.80f},
            {"beta", 0.50f, 0.99f},
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
        "datasets/model/equity/cev/parameters/cev_01.json";
    write_model_dataset(
        "cev_01",
        "CEV",
        dataset,
        "catalog/model/equity/cev/parameters/cev_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/cev/parameters/cev_01.json",
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"sigma", "Local-volatility scale."},
            {"beta", "CEV elasticity in [0.5, 1)."},
        },
        {
            {"sde", "dS = (r-q) S dt + sigma S^beta dW."},
            {
                "simulation",
                "Milstein time stepping with absorption at zero."
            },
            {
                "boundary",
                "Negative numerical proposals are projected to the "
                "attainable absorbing boundary."
            },
        },
        rows
    );
    validate_model_dataset_file(dataset);
}
