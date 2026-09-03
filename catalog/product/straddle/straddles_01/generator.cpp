// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the straddle dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/straddle/straddles_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/straddle/straddles_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/straddles/"
        "straddles_01.json";

    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_business_day_grid(21U, 756U, 45U),
        20U,
        0.2f,
        linear_business_day_grid(5U, 1764U, 10U),
        10U,
        0.2f
    );

    write_product_dataset(
        "straddles_01",
        "Straddles",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Common call and put strike in normalized spot units."},
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "abs(S_T - K)"},
            {"decomposition", "European call plus European put."},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
