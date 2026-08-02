// Generate reproducible Hull-White one-factor model parameters.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"
#include "tools/datasets/ornstein_uhlenbeck_generation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

// Generate the Hull-White dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/hull_white/hull_white_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/hull_white/hull_white_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/"
        "hull_white/hull_white_01.json";

    constexpr std::uint64_t seed = 730000201ULL;
    const GeneratedRows rows =
        ornstein_uhlenbeck::generate_dynamics_rows(
            1'000U,
            seed,
            {
                {0.03f, 1.0f},
                {0.0025f, 0.025f},
            }
        );

    write_model_dataset(
        "hull_white_01",
        "Hull-White one-factor",
        dataset_path,
        catalog_path,
        url,
        {
            {"mean_reversion", "Positive mean-reversion speed a."},
            {"volatility", "Non-negative short-rate volatility sigma."},
        },
        {
            {"state", "dx_t = -a x_t dt + sigma dW_t"},
            {"short_rate", "r_t = x_t + phi(t)"},
            {
                "transition",
                "x and its integral are simulated jointly and exactly; "
                "the curve shift is integrated analytically"
            },
            {
                "curve_fit",
                "phi(t) is determined by the initial instantaneous "
                "forward curve f(0,t)"
            },
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
