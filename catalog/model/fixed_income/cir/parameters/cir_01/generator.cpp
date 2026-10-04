// Generate CIR rows by sampling ordinary rate parameters first, then sigma
// from row-dependent bounds that control the Feller-ratio range.
#include "tools/datasets/parameter_dataset.hpp"
#include "tools/sampling/parameters/cir_generation.hpp"
#include "common/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

// Generate the CIR dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/fixed_income/cir/parameters/cir_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/fixed_income/cir/parameters/cir_01/generation.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/"
        "fixed_income/cir/parameters/cir_01.json";

    constexpr std::uint64_t seed = 770000201ULL;
    const GeneratedRows rows = cir::generate_core_stress_rows(seed);

    write_model_dataset(
        "cir_01",
        "CIR short rate",
        dataset_path,
        catalog_path,
        url,
        {
            {"mean_reversion", "Positive mean-reversion speed kappa."},
            {"long_term_mean", "Positive long-run short-rate mean theta."},
            {"volatility", "Positive square-root diffusion scale sigma."},
            {"initial_state", "Non-negative initial short rate r(0)."},
        },
        {
            {
                "short_rate",
                "dr_t = kappa * (theta - r_t) dt + sigma * sqrt(r_t) dW_t"
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
}
