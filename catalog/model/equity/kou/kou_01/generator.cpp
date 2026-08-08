// Generate ordered 90/10 Kou double-exponential parameter rows.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>

int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint64_t seed = 710000701ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"volatility", 0.08f, 0.45f},
            {"jump_intensity", 0.02f, 1.0f},
            {"up_probability", 0.20f, 0.70f},
            {"positive_jump_rate", 3.0f, 20.0f},
            {"negative_jump_rate", 2.0f, 20.0f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"volatility", 0.03f, 0.80f},
            {"jump_intensity", 0.001f, 3.0f},
            {"up_probability", 0.03f, 0.97f},
            {"positive_jump_rate", 2.10f, 35.0f},
            {"negative_jump_rate", 0.50f, 35.0f},
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
        "datasets/model/equity/kou/kou_01.json";
    write_model_dataset(
        "kou_01",
        "Kou double-exponential jump diffusion",
        dataset,
        "catalog/model/equity/kou/kou_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/kou/kou_01.json",
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"volatility", "Diffusion volatility."},
            {"jump_intensity", "Poisson jump intensity."},
            {"up_probability", "Probability of an upward jump."},
            {
                "positive_jump_rate",
                "Exponential rate of positive log jumps; greater than two "
                "for finite payoff variance."
            },
            {
                "negative_jump_rate",
                "Exponential rate of negative log-jump magnitudes."
            },
        },
        {
            {
                "log_spot",
                "Brownian diffusion plus compound-Poisson double-"
                "exponential jumps."
            },
            {
                "martingale_correction",
                "Uses the first exponential moment of the asymmetric jump "
                "multiplier."
            },
            {
                "simulation",
                "Exact independent increments with a path-local variable-"
                "length uniform stream."
            },
        },
        rows
    );
    validate_model_dataset_file(dataset);
}
