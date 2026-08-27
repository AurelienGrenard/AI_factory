#include "common/dataset_validation.hpp"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    constexpr std::uint64_t seed = 710001801ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"initial_feedback", 0.03f, 0.20f},
            {"quadratic_scale", 0.10f, 0.80f},
            {"quadratic_shift", 0.02f, 0.20f},
            {"variance_floor", 0.0005f, 0.02f},
            {"feedback_rate", 0.30f, 3.0f},
            {"feedback_volatility", 0.30f, 2.0f},
            {"hurst_exponent", 0.01f, 0.20f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"initial_feedback", -0.30f, 0.50f},
            {"quadratic_scale", 0.02f, 2.0f},
            {"quadratic_shift", -0.20f, 0.50f},
            {"variance_floor", 0.0001f, 0.10f},
            {"feedback_rate", 0.05f, 6.0f},
            {"feedback_volatility", 0.05f, 4.0f},
            {"hurst_exponent", 0.005f, 0.45f},
        })
    );
    for (auto& row : rows.rows) row["spot"] = 1.0f;
    const std::filesystem::path dataset =
        "datasets/model/equity/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json";
    write_model_dataset(
        "quadratic_rough_heston_01", "quadratic rough Heston", dataset,
        "catalog/model/equity/quadratic_rough_heston/parameters/quadratic_rough_heston_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json",
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
