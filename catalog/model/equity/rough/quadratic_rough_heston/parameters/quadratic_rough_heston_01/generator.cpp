#include "common/dataset_validation.hpp"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    constexpr std::uint64_t seed = 1610001801ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"initial_feedback", 0.02f, 0.15f},
            {"quadratic_scale", 0.02f, 0.12f},
            {"quadratic_shift", 0.02f, 0.15f},
            {"variance_floor", 0.0005f, 0.015f},
            {"feedback_rate", 0.1f, 0.8f},
            {"feedback_volatility", 0.1f, 0.6f},
            {"hurst_exponent", 0.05f, 0.2f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"initial_feedback", -0.1f, 0.25f},
            {"quadratic_scale", 0.01f, 0.18f},
            {"quadratic_shift", -0.05f, 0.2f},
            {"variance_floor", 0.0001f, 0.04f},
            {"feedback_rate", 0.05f, 1.2f},
            {"feedback_volatility", 0.05f, 0.8f},
            {"hurst_exponent", 0.02f, 0.3f},
        })
    );
    for (auto& row : rows.rows) row["spot"] = 1.0f;
    rows.construction["aligned_mc_domain"] = {
        {"quadratic_feedback", "bounded a, lambda and eta for long-maturity aligned Monte Carlo prices"},
        {"qualification", "European parity and tail checks required before use as reference"},
    };
    const std::filesystem::path dataset =
        "datasets/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json";
    write_model_dataset(
        "quadratic_rough_heston_01", "quadratic rough Heston", dataset,
        "catalog/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json",
        {
            {"spot", "Initial spot; fixed to 1."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_feedback", "Initial feedback factor Z0."},
            {"quadratic_scale", "Positive coefficient a."},
            {"quadratic_shift", "Quadratic center b."},
            {"variance_floor", "Strictly positive floor c."},
            {"feedback_rate", "Feedback rate lambda."},
            {"feedback_volatility", "Feedback volatility eta."},
            {"hurst_exponent", "Fractional roughness H."},
        },
        {
            {"variance", "V=a(Z-b)^2+c."},
            {"feedback", "Z=Z0-lambda K*Z dt+lambda eta K*sqrt(V)dW."},
            {"spot", "dS/S=(r-q)dt+sqrt(V)dW with the same W."},
            {"simulation", "Positive exponential N-factor lift."},
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::quadratic_rough_heston::load_models(dataset).size()
        != 1000U) {
        throw std::runtime_error("Quadratic rough-Heston reload failed.");
    }
}
