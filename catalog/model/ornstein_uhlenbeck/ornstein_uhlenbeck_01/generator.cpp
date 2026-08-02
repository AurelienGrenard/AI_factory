// Generate reproducible Ornstein-Uhlenbeck short-rate parameters.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"
#include "tools/datasets/ornstein_uhlenbeck_generation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

// Generate the Ornstein-Uhlenbeck dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/ornstein_uhlenbeck/ornstein_uhlenbeck_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/ornstein_uhlenbeck/ornstein_uhlenbeck_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/"
        "ornstein_uhlenbeck/ornstein_uhlenbeck_01.json";

    constexpr std::uint64_t seed = 740000201ULL;
    const GeneratedRows rows = ornstein_uhlenbeck::generate_rows(
        1'000U,
        seed,
        {
            {
                {0.03f, 1.0f},
                {0.0025f, 0.025f},
            },
            {0.0f, 0.08f},
        }
    );

    write_model_dataset(
        "ornstein_uhlenbeck_01",
        "Ornstein-Uhlenbeck short rate",
        dataset_path,
        catalog_path,
        url,
        {
            {"mean_reversion", "Positive mean-reversion speed a."},
            {"volatility", "Instantaneous short-rate volatility sigma."},
            {"initial_state", "Initial short rate x(0)."},
        },
        {
            {"state", "dx_t = -a x_t dt + sigma dW_t"},
            {"short_rate", "r_t = x_t"},
            {
                "transition",
                "exact Gaussian state and optional joint state-integral transitions"
            },
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
