// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the Asian-option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/asian_options/asian_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/asian_options/asian_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/asian_options/"
        "asian_options_01.json";

    constexpr float log_moneyness_slope = 0.15f;
    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_business_day_grid(21U, 756U, 45U),
        20U,
        log_moneyness_slope,
        linear_business_day_grid(5U, 1764U, 10U),
        10U,
        log_moneyness_slope
    );
    // Preserve the exact lower maturity bound in human-readable metadata.

    write_product_dataset(
        "asian_options_01",
        "Asian Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "max(side * (arithmetic_mean(S_[0,T]) - K), 0), side = +1 call / -1 put"},
            {"monitoring", "Simulation-grid endpoints including 0 and T."},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
