#include "common/dataset_validation.hpp"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    constexpr std::uint64_t seed = 1610001301ULL;
    constexpr float spot_min = 0.20f;
    constexpr float spot_max = 5.0f;
    constexpr float core_spot_min = 0.25f;
    constexpr float core_spot_max = 4.0f;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"spot", core_spot_min, core_spot_max},
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"initial_volatility", 0.08f, 0.50f},
            {"volatility_of_volatility", 0.10f, 1.50f},
            {"rho", -0.90f, -0.20f},
            {"beta", 0.50f, 0.95f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"spot", spot_min, spot_max},
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"initial_volatility", 0.04f, 0.55f},
            {"volatility_of_volatility", 0.08f, 1.80f},
            {"rho", -0.98f, -0.10f},
            {"beta", 0.40f, 0.95f},
        })
    );
    rows.construction["aligned_mc_domain"] = {
        {"rho", "negative spot/alpha correlation for the aligned price reference campaign"},
        {"beta", "CEV elasticity at least 0.50 in core and 0.40 in stress, away from the absorption-sensitive beta-near-zero regime"},
        {"stress_volatility", "initial volatility at most 0.55 and lognormal alpha volatility at most 1.80"},
    };
    rows.construction["spot_range"] = {
        {"S0_min", spot_min}, {"S0_max", spot_max}
    };
    rows.construction["parameterization"] = {
        {"initial_volatility", "initial log-return volatility sigma_0"},
        {"dimensional_alpha", "alpha_0=sigma_0*S0^(1-beta)"},
    };
    const std::filesystem::path dataset =
        "datasets/model/equity/markovian/sabr/parameters/sabr_01.json";
    write_model_dataset(
        "sabr_01", "SABR equity", dataset,
        "catalog/model/equity/markovian/sabr/parameters/sabr_01/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/markovian/sabr/parameters/sabr_01.json",
        {
            {"spot", "Initial equity spot, sampled between S0_min and S0_max."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_volatility", "Initial log-return volatility at S0."},
            {"volatility_of_volatility", "Lognormal alpha volatility."},
            {"rho", "Spot/alpha Brownian correlation."},
            {"beta", "CEV elasticity."},
        },
        {
            {"spot", "dS_t=(r-q)S_t dt+alpha_t S_t^beta dW_t."},
            {"alpha", "dalpha_t=nu alpha_t dZ_t."},
            {"correlation", "d<W,Z>_t=rho dt."},
            {"simulation", "Lamperti step for S and exact lognormal alpha endpoint."},
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::sabr::load_models(dataset).size() != 1000U) {
        throw std::runtime_error("SABR generator reload failed.");
    }
}
