// Generate ordered 90/10 flat-xi0 rough-Bergomi parameter rows.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <utility>

int main() {
    using namespace ai_factory::workbench::datasets;

    constexpr std::uint64_t seed = 710001001ULL;
    constexpr float fixed_spot = 1.0f;
    constexpr float fixed_xi_0 = 0.04f;

    GeneratedRows core = uniform_rows(900U, seed, {
        {"risk_free_rate", 0.001f, 0.08f},
        {"dividend_yield", 0.0f, 0.06f},
        {"eta", 0.50f, 3.0f},
        {"hurst_exponent", 0.03f, 0.25f},
        {"rho", -0.95f, -0.30f},
    });
    GeneratedRows stress = uniform_rows(100U, seed + 1U, {
        {"risk_free_rate", 0.0001f, 0.12f},
        {"dividend_yield", 0.0f, 0.10f},
        {"eta", 0.10f, 5.0f},
        {"hurst_exponent", 0.01f, 0.45f},
        {"rho", -0.99f, 0.20f},
    });
    GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );
    for (ParameterRow& row : rows.rows) {
        ParameterRow completed = {
            {"spot", fixed_spot},
            {"xi_0", fixed_xi_0},
        };
        for (const auto& [name, value] : row.items()) {
            completed[name] = value;
        }
        row = std::move(completed);
    }
    rows.construction["fixed_parameters"] = {
        {"spot", fixed_spot},
        {"xi_0", fixed_xi_0},
        {"forward_variance_curve", "xi_0(t) = xi_0 for all t"},
    };
    rows.construction["rough_driver_discretization"] = {
        {"scheme", "Bennedsen-Lunde-Pakkanen hybrid scheme"},
        {"kappa", 1},
        {"far_cell_evaluation", "L2-optimal b_k^*"},
    };

    const std::filesystem::path dataset =
        "datasets/model/equity/rough_bergomi/parameters/rough_bergomi_01.json";
    write_model_dataset(
        "rough_bergomi_01",
        "rough Bergomi",
        dataset,
        "catalog/model/equity/rough_bergomi/parameters/rough_bergomi_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/"
        "rough_bergomi/parameters/rough_bergomi_01.json",
        {
            {"spot", "Initial spot; fixed to 1 in this dataset."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {
                "xi_0",
                "Flat forward variance level; fixed to 0.04 in this dataset."
            },
            {"eta", "Volatility of the logarithmic variance."},
            {
                "hurst_exponent",
                "Roughness exponent H, strictly between 0 and 0.5."
            },
            {"rho", "Spot/rough-driver Brownian correlation."},
        },
        {
            {
                "spot",
                "dS_t / S_t = (r-q) dt + sqrt(v_t) dZ_t."
            },
            {
                "variance",
                "v_t = xi_0 exp(eta Y_t - eta^2 t^(2H) / 2)."
            },
            {
                "rough_driver",
                "Y_t = sqrt(2H) integral_0^t (t-s)^(H-1/2) dW_s."
            },
            {
                "correlation",
                "d<Z,W>_t = rho dt."
            },
            {
                "simulation",
                "Hybrid scheme with kappa=1 and optimal far-cell weights."
            },
        },
        rows
    );
    validate_model_dataset_file(dataset);
}
