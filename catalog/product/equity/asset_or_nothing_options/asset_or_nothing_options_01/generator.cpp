// Generate asset-or-nothing options on the standard equity grid.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the asset-or-nothing dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/asset_or_nothing_options/"
        "asset_or_nothing_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/asset_or_nothing_options/"
        "asset_or_nothing_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/"
        "asset_or_nothing_options/asset_or_nothing_options_01.json";

    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, 45U),
        20U,
        0.2f,
        linear_grid(1.0f / 52.0f, 7.0f, 10U),
        10U,
        0.2f
    );

    write_product_dataset(
        "asset_or_nothing_options_01",
        "Asset-or-Nothing Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Trigger strike in normalized spot units."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "S_T * 1{side * (S_T - K) > 0}, side = +1 call / -1 put"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
