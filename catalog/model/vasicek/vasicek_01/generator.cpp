// Generate reproducible Vasicek short-rate parameters.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"
#include "tools/datasets/vasicek_generation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

// Generate the Vasicek dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/vasicek/vasicek_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/vasicek/vasicek_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/"
        "vasicek/vasicek_01.json";

    constexpr std::uint64_t seed = 750000201ULL;
    const GeneratedRows rows = vasicek::generate_rows(
        1'000U,
        seed,
        {
            {
                {0.03f, 1.0f},
                {0.0f, 0.08f},
                {0.0025f, 0.025f},
            },
            {0.0f, 0.08f},
        }
    );

    write_model_dataset(
        "vasicek_01",
        "Vasicek short rate",
        dataset_path,
        catalog_path,
        url,
        {
            {"mean_reversion", "Positive mean-reversion speed a."},
            {"long_term_mean", "Long-run short-rate mean b."},
            {"volatility", "Instantaneous short-rate volatility sigma."},
            {"initial_state", "Initial short rate r(0)."},
        },
        {
            {"short_rate", "dr_t = a * (b - r_t) dt + sigma dW_t"},
            {
                "transition",
                "exact Gaussian state and optional joint state-integral transitions"
            },
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
