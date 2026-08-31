#include "common/dataset_validation.hpp"
#include "model/equity/rough/rough_stein_stein/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    constexpr std::uint64_t seed = 710001701ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"volatility_level", 0.10f, 0.35f},
            {"mean_reversion", 0.2f, 4.0f},
            {"volatility_of_volatility", 0.05f, 0.50f},
            {"hurst_exponent", 0.03f, 0.25f},
            {"rho", -0.90f, 0.10f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"volatility_level", 0.0f, 0.80f},
            {"mean_reversion", 0.0f, 8.0f},
            {"volatility_of_volatility", 0.01f, 1.5f},
            {"hurst_exponent", 0.01f, 0.45f},
            {"rho", -0.99f, 0.80f},
        })
    );
    for (auto& row : rows.rows) row["spot"] = 1.0f;
    const std::filesystem::path dataset =
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json";
    write_model_dataset(
        "rough_stein_stein_01", "rough Stein-Stein", dataset,
        "catalog/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json",
        {
            {"spot", "Initial spot; fixed to 1."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"volatility_level", "Constant mean arithmetic volatility."},
            {"mean_reversion", "Linear Volterra mean-reversion rate."},
            {"volatility_of_volatility", "Gaussian Volterra scale."},
            {"hurst_exponent", "Fractional roughness H."},
            {"rho", "Spot/volatility Brownian correlation."},
        },
        {
            {"volatility", "X=theta-kappa K*(X-theta)dt+nu K*dW."},
            {"spot", "dS/S=(r-q)dt+X dB."},
            {"simulation", "Linear resolvent followed by one hybrid FFT."},
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::rough_stein_stein::load_models(dataset).size() != 1000U) {
        throw std::runtime_error("Rough Stein-Stein reload failed.");
    }
}
