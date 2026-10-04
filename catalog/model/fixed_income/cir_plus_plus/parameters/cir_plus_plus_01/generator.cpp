// Generate CIR++ factor rows with the shared CIR core/stress law and metadata.
#include "tools/datasets/parameter_dataset.hpp"
#include "tools/sampling/parameters/cir_generation.hpp"
#include "common/dataset_validation.hpp"
#include "model/fixed_income/cir_plus_plus/dataset.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

// Generate the CIR++ factor dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/fixed_income/cir_plus_plus/parameters/cir_plus_plus_01/generation.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/"
        "fixed_income/cir_plus_plus/parameters/cir_plus_plus_01.json";

    constexpr std::uint64_t seed = 770000801ULL;
    const GeneratedRows rows = cir::generate_core_stress_rows(seed);

    write_model_dataset(
        "cir_plus_plus_01",
        "CIR++ shifted short rate",
        dataset_path,
        catalog_path,
        url,
        {
            {"mean_reversion", "Positive mean-reversion speed kappa."},
            {"long_term_mean", "Positive long-run factor mean theta."},
            {"volatility", "Positive square-root diffusion scale sigma."},
            {"initial_state", "Non-negative initial CIR factor y(0), not the fitted short rate."},
        },
        {
            {"short_rate", "r(t) = y(t) + phi(t)"},
            {"shift", "phi(t) = f_market(0,t) - f_CIR(0,t)"},
            {
                "factor",
                "dy_t = kappa * (theta - y_t) dt + sigma * sqrt(y_t) dW_t"
            },
            {
                "transition",
                "exact non-central chi-square endpoint transition"
            },
            {
                "boundary",
                "the exact transition supports both sides of the Feller threshold"
            },
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
    static_cast<void>(ai_factory::workbench::model::fixed_income::cir_plus_plus::load_models(dataset_path));
}
