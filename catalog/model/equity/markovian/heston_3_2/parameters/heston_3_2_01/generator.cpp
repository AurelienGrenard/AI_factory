#include "common/dataset_validation.hpp"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    constexpr std::uint64_t seed = 710001401ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"initial_variance", 0.01f, 0.09f},
            {"mean_reversion", 5.0f, 40.0f},
            {"long_run_variance", 0.015f, 0.09f},
            {"volatility_of_variance", 1.0f, 8.0f},
            {"rho", -0.95f, -0.10f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"initial_variance", 0.0025f, 0.25f},
            {"mean_reversion", 0.5f, 80.0f},
            {"long_run_variance", 0.0025f, 0.25f},
            {"volatility_of_variance", 0.2f, 15.0f},
            {"rho", -0.99f, 0.50f},
        })
    );
    for (auto& row : rows.rows) row["spot"] = 1.0f;
    const std::filesystem::path dataset =
        "datasets/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json";
    write_model_dataset(
        "heston_3_2_01", "Heston 3/2", dataset,
        "catalog/model/equity/markovian/heston_3_2/parameters/heston_3_2_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json",
        {
            {"spot", "Initial spot; fixed to 1."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_variance", "Initial instantaneous variance."},
            {"mean_reversion", "3/2 variance mean-reversion coefficient."},
            {"long_run_variance", "3/2 variance equilibrium level."},
            {"volatility_of_variance", "Coefficient of V^(3/2)dW."},
            {"rho", "Spot/variance Brownian correlation."},
        },
        {
            {"spot", "dS/S=(r-q)dt+sqrt(V)dW_S."},
            {"variance", "dV=kappa V(theta-V)dt+epsilon V^(3/2)dW_V."},
            {"simulation", "Milstein stepping of reciprocal CIR U=1/V."},
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::heston_3_2::load_models(dataset).size() != 1000U) {
        throw std::runtime_error("Heston 3/2 generator reload failed.");
    }
}
