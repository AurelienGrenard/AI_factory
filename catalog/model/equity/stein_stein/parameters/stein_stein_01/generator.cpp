#include "common/dataset_validation.hpp"
#include "model/equity/markovian/stein_stein/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    constexpr std::uint64_t seed = 710001501ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"initial_volatility", 0.10f, 0.35f},
            {"mean_reversion", 0.5f, 8.0f},
            {"volatility_of_volatility", 0.05f, 0.50f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"initial_volatility", 0.0f, 0.80f},
            {"mean_reversion", 0.05f, 15.0f},
            {"volatility_of_volatility", 0.01f, 1.5f},
        })
    );
    for (auto& row : rows.rows) {
        row["spot"] = 1.0f;
        row["rho"] = 0.0f;
    }
    rows.construction["fixed_parameters"] = {{"spot", 1.0f}, {"rho", 0.0f}};
    const std::filesystem::path dataset =
        "datasets/model/equity/stein_stein/parameters/stein_stein_01.json";
    write_model_dataset(
        "stein_stein_01", "Stein-Stein", dataset,
        "catalog/model/equity/stein_stein/parameters/stein_stein_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/stein_stein/parameters/stein_stein_01.json",
        {
            {"spot", "Initial spot; fixed to 1."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_volatility", "Initial arithmetic volatility."},
            {"mean_reversion", "Zero-mean OU rate."},
            {"volatility_of_volatility", "OU diffusion coefficient."},
            {"rho", "Fixed to zero for the original Stein-Stein model."},
        },
        {
            {"spot", "dS/S=(r-q)dt+X dW_S."},
            {"volatility", "dX=-kappa X dt+nu dW_X."},
            {"simulation", "Exact OU endpoint coupled to the Euler stock increment."},
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::stein_stein::load_models(dataset).size() != 1000U) {
        throw std::runtime_error("Stein-Stein generator reload failed.");
    }
}
