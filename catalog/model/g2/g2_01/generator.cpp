// Generate reproducible standalone G2 short-rate parameters.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"
#include "tools/datasets/g2_generation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>

// Generate the G2 dataset and adjacent catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/g2/g2_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/g2/g2_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/g2/g2_01.json";

    constexpr std::uint64_t seed = 750000201ULL;
    const GeneratedRows rows = g2::generate_rows(
        1'000U,
        seed,
        {
            {
                {0.03f, 0.35f},
                {0.10f, 1.00f},
                {0.0025f, 0.018f},
                {0.0015f, 0.012f},
                {-0.75f, 0.25f},
            },
            {0.0f, 0.05f},
            {0.0f, 0.03f},
        }
    );

    write_model_dataset(
        "g2_01",
        "G2 Gaussian two-factor short rate",
        dataset_path,
        catalog_path,
        url,
        {
            {"mean_reversion_x", "Positive mean-reversion speed a."},
            {"volatility_x", "Non-negative first-factor volatility sigma."},
            {"mean_reversion_y", "Positive mean-reversion speed b above a."},
            {"volatility_y", "Non-negative second-factor volatility eta."},
            {"correlation", "Brownian correlation rho in [-1, 1]."},
            {"initial_state_x", "Initial first Gaussian factor x(0)."},
            {"initial_state_y", "Initial second Gaussian factor y(0)."},
        },
        {
            {"state_x", "dx_t = -a x_t dt + sigma dW_t^x"},
            {"state_y", "dy_t = -b y_t dt + eta dW_t^y"},
            {"correlation", "d<W^x,W^y>_t = rho dt"},
            {"short_rate", "r_t = x_t + y_t"},
            {"initial_short_rate", "r(0) = x(0) + y(0)"},
            {"transition", "exact correlated Gaussian transitions"},
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
