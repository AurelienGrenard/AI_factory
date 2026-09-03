// Generate ordered 90/10 rough-SABR parameter rows.
#include "model/equity/rough/rough_sabr/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <utility>

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::datasets;

    constexpr std::uint64_t seed = 710001201ULL;
    constexpr float spot_min = 0.05f;
    constexpr float spot_max = 10.0f;
    constexpr float core_spot_min = 0.25f;
    constexpr float core_spot_max = 4.0f;
    constexpr float fixed_xi_0 = 0.04f;

    GeneratedRows core = uniform_rows(900U, seed, {
        {"spot", core_spot_min, core_spot_max},
        {"risk_free_rate", 0.001f, 0.08f},
        {"dividend_yield", 0.0f, 0.06f},
        {"eta", 0.50f, 3.0f},
        {"hurst_exponent", 0.03f, 0.25f},
        {"rho", -0.95f, -0.30f},
        {"beta", 0.70f, 1.0f},
    });
    GeneratedRows stress = uniform_rows(100U, seed + 1U, {
        {"spot", spot_min, spot_max},
        {"risk_free_rate", -0.03f, 0.12f},
        {"dividend_yield", 0.0f, 0.10f},
        {"eta", 0.10f, 5.0f},
        {"hurst_exponent", 0.01f, 0.45f},
        {"rho", -0.99f, 0.20f},
        {"beta", 0.50f, 1.0f},
    });
    GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );
    for (ParameterRow& row : rows.rows) {
        ParameterRow completed = {
            {"spot", row.at("spot")},
            {"risk_free_rate", row.at("risk_free_rate")},
            {"dividend_yield", row.at("dividend_yield")},
            {"xi_0", fixed_xi_0},
            {"eta", row.at("eta")},
            {"hurst_exponent", row.at("hurst_exponent")},
            {"rho", row.at("rho")},
            {"beta", row.at("beta")},
        };
        row = std::move(completed);
    }
    rows.construction["fixed_parameters"] = {
        {"xi_0", fixed_xi_0},
        {"forward_variance_curve", "xi_0(t) = xi_0 for all t"},
    };
    rows.construction["spot_range"] = {
        {"S0_min", spot_min}, {"S0_max", spot_max}
    };
    rows.construction["parameterization"] = {
        {"initial_log_return_volatility", "sqrt(xi_0)"},
        {"dimensional_alpha_0", "sqrt(xi_0)*S0^(1-beta)"},
    };
    rows.construction["rough_driver_discretization"] = {
        {"scheme", "Bennedsen-Lunde-Pakkanen hybrid scheme"},
        {"kappa", 1},
        {"far_cell_evaluation", "L2-optimal b_k^*"},
    };

    const std::filesystem::path dataset =
        "datasets/model/equity/rough/rough_sabr/parameters/rough_sabr_01.json";
    write_model_dataset(
        "rough_sabr_01",
        "rough SABR",
        dataset,
        "catalog/model/equity/rough/rough_sabr/parameters/rough_sabr_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/"
        "rough_sabr/parameters/rough_sabr_01.json",
        {
            {"spot", "Initial spot, sampled between S0_min and S0_max."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {
                "xi_0",
                "Flat initial squared-volatility level; fixed to 0.04."
            },
            {"eta", "Volatility of the logarithmic squared volatility."},
            {
                "hurst_exponent",
                "Roughness exponent H, strictly between 0 and 0.5."
            },
            {"rho", "Spot/rough-driver Brownian correlation."},
            {"beta", "CEV elasticity, between 0.5 and 1."},
        },
        {
            {
                "rough_driver",
                "Y_t = sqrt(2H) integral_0^t (t-s)^(H-1/2) dW_s."
            },
            {
                "volatility",
                "alpha_t = sqrt(xi_0) S0^(1-beta) "
                "exp(eta Y_t / 2 - eta^2 t^(2H) / 4)."
            },
            {
                "spot",
                "dS_t = (r-q) S_t dt + alpha_t S_t^beta dZ_t."
            },
            {"correlation", "d<Z,W>_t = rho dt."},
            {
                "simulation",
                "Hybrid rough convolution followed by Lamperti spot steps."
            },
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::rough_sabr::load_models(dataset).size() != 1000U) {
        throw std::runtime_error(
            "The generated rough-SABR dataset did not reload 1000 rows."
        );
    }
}
