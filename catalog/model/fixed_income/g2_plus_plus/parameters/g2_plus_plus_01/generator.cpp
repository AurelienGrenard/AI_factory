// Generate reproducible curve-independent G2++ parameters.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"
#include "tools/datasets/g2_generation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

// Generate the G2++ dataset and adjacent catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json";
    const std::filesystem::path catalog_path =
        "catalog/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/model/"
        "fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json";

    constexpr std::uint64_t seed = 760000201ULL;
    GeneratedRows core = g2::generate_process_rows(
        900U,
        seed,
        {
            {0.03f, 0.35f},
            {0.10f, 1.00f},
            {0.0025f, 0.018f},
            {0.0015f, 0.012f},
            {-0.75f, 0.25f},
        }
    );
    GeneratedRows stress = g2::generate_process_rows(
        100U,
        seed + 2ULL,
        {
            {0.005f, 0.70f},
            {0.02f, 1.80f},
            {0.001f, 0.035f},
            {0.001f, 0.025f},
            {-0.98f, 0.75f},
        }
    );
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_model_dataset(
        "g2_plus_plus_01",
        "G2++ Gaussian two-factor",
        dataset_path,
        catalog_path,
        url,
        {
            {"mean_reversion_x", "Positive mean-reversion speed a."},
            {"volatility_x", "Non-negative first-factor volatility sigma."},
            {"mean_reversion_y", "Positive mean-reversion speed b above a."},
            {"volatility_y", "Non-negative second-factor volatility eta."},
            {"correlation", "Brownian correlation rho in [-1, 1]."},
        },
        {
            {"state_x", "dx_t = -a x_t dt + sigma dW_t^x"},
            {"state_y", "dy_t = -b y_t dt + eta dW_t^y"},
            {"correlation", "d<W^x,W^y>_t = rho dt"},
            {"short_rate", "r_t = x_t + y_t + phi(t)"},
            {"curve_fit", "phi(t) fits the supplied initial forward curve"},
        },
        rows
    );
    validate_model_dataset_file(dataset_path);
}
