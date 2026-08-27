// Generate risk-neutral Black-Scholes rows in ordered core and stress regimes.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/equity/black_scholes/parameters/black_scholes_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/equity/black_scholes/parameters/black_scholes_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/equity/"
        "black_scholes/parameters/black_scholes_01.json";
    constexpr std::uint64_t seed = 710000101ULL;

    GeneratedRows core = uniform_rows(
        900U,
        seed,
        {
            {"spot", 1.0f, 1.0f},
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"volatility", 0.08f, 0.45f},
        }
    );
    core.construction["seed"] = seed;
    GeneratedRows stress = uniform_rows(
        100U,
        seed + 1ULL,
        {
            {"spot", 1.0f, 1.0f},
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"volatility", 0.03f, 0.80f},
        }
    );
    stress.construction["seed"] = seed + 1ULL;
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_model_dataset(
        "black_scholes_01",
        "Black-Scholes",
        dataset_path,
        catalog_path,
        url,
        {
            {"spot", "Initial spot."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"volatility", "Constant annualized spot volatility."},
        },
        {
            {
                "log_spot",
                "log(S_t) = log(S_0) + (r-q-sigma^2/2)t + sigma W_t"
            },
            {"transition", "Exact Gaussian log-price increment on every interval."},
            {"measure", "Risk-neutral measure with deterministic r and q."},
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
