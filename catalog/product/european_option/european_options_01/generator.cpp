// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the European-option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/european_option/european_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/european_option/european_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/european_options/"
        "european_options_01.json";

    constexpr float log_moneyness_slope = 0.2f;
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
        "european_options_01",
        "European Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "max(side * (S_T - K), 0), side = +1 call / -1 put"},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
