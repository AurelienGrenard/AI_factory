// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the Geometric-Asian-call dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/geometric_asian_calls/geometric_asian_calls_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/geometric_asian_calls/geometric_asian_calls_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/geometric_asian_calls/"
        "geometric_asian_calls_01.json";

    constexpr std::size_t maturity_count = 50U;
    constexpr std::size_t strikes_per_maturity = 20U;
    constexpr float log_moneyness_slope = 0.15f;
    GeneratedRows rows = maturity_dependent_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, maturity_count),
        strikes_per_maturity,
        log_moneyness_slope
    );
    // Preserve the exact lower maturity bound in human-readable metadata.
    rows.construction["grid"]["maturity"]["minimum"] = "1 / 12";

    write_product_dataset(
        "geometric_asian_calls_01",
        "Geometric Asian Calls",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "max(geometric_mean(S_[0,T]) - K, 0)"},
            {"monitoring", "Simulation-grid endpoints including 0 and T."},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
