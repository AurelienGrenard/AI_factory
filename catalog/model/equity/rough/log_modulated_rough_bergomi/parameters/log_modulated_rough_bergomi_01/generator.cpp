#include "common/dataset_validation.hpp"
#include "model/equity/rough/log_modulated_rough_bergomi/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    constexpr std::uint64_t seed = 710001601ULL;
    GeneratedRows rows = core_stress_rows(
        uniform_rows(900U, seed, {
            {"risk_free_rate", 0.001f, 0.08f},
            {"dividend_yield", 0.0f, 0.06f},
            {"xi_0", 0.01f, 0.09f},
            {"eta", 0.50f, 3.0f},
            {"hurst_exponent", 0.01f, 0.20f},
            {"rho", -0.95f, -0.20f},
            {"log_modulation_scale", 0.03f, 0.30f},
            {"log_modulation_power", 1.20f, 4.0f},
        }),
        uniform_rows(100U, seed + 1U, {
            {"risk_free_rate", -0.03f, 0.12f},
            {"dividend_yield", 0.0f, 0.10f},
            {"xi_0", 0.0025f, 0.25f},
            {"eta", 0.10f, 5.0f},
            {"hurst_exponent", 0.0f, 0.45f},
            {"rho", -0.99f, 0.50f},
            {"log_modulation_scale", 0.005f, 1.0f},
            {"log_modulation_power", 1.01f, 8.0f},
        })
    );
    for (auto& row : rows.rows) row["spot"] = 1.0f;
    const std::filesystem::path dataset =
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json";
    write_model_dataset(
        "log_modulated_rough_bergomi_01", "log-modulated rough Bergomi",
        dataset,
        "catalog/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01/dataset.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json",
        {
            {"spot", "Initial spot; fixed to 1."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"xi_0", "Flat initial forward variance."},
            {"eta", "Log-variance volatility."},
            {"hurst_exponent", "Power component H in [0, 1/2)."},
            {"rho", "Spot/Volterra Brownian correlation."},
            {"log_modulation_scale", "Positive logarithmic scale zeta."},
            {"log_modulation_power", "Logarithmic power p>1."},
        },
        {
            {"kernel", "C t^(H-1/2) max(zeta log(1/t),1)^(-p)."},
            {"variance", "v=xi_0 exp(eta Y-eta^2 Var(Y)/2)."},
            {"simulation", "Normalized arbitrary-kernel hybrid FFT."},
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::log_modulated_rough_bergomi::load_models(dataset).size()
        != 1000U) {
        throw std::runtime_error("Log-modulated rough-Bergomi reload failed.");
    }
}
