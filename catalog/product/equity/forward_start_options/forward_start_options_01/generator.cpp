// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the Forward-start-option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/forward_start_options/forward_start_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/forward_start_options/forward_start_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/forward_start_options/"
        "forward_start_options_01.json";

    constexpr float log_moneyness_slope = 0.2f;
    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_business_day_grid(21U, 756U, 45U),
        20U,
        log_moneyness_slope,
        linear_business_day_grid(5U, 1764U, 10U),
        10U,
        log_moneyness_slope
    );
    for (auto& row : rows.rows) {
        const std::uint32_t maturity =
            row.at("maturity").get<std::uint32_t>();
        row["moneyness"] = row.at("strike");
        row["reset_time"] = maturity / 2U;
        row.erase("strike");
    }
    // Preserve the exact lower maturity bound in human-readable metadata.
    rows.construction["grid"]["moneyness"] =
        rows.construction["grid"].at("strike");
    rows.construction["grid"].erase("strike");
    rows.construction["reset_time"] = {
        {"rule", "floor(maturity / 2) business days"},
    };

    write_product_dataset(
        "forward_start_options_01",
        "Forward-Start Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"moneyness", "Strike multiplier applied to the reset spot."},
            {"reset_time", "Business day when the floating strike is fixed."},
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "max(side * (S_T - moneyness * S_reset), 0), side = +1 call / -1 put"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
