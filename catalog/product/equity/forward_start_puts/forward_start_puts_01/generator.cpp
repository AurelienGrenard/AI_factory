// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the Forward-start-put dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/forward_start_puts/forward_start_puts_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/forward_start_puts/forward_start_puts_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/forward_start_puts/"
        "forward_start_puts_01.json";

    constexpr std::size_t maturity_count = 50U;
    constexpr std::size_t strikes_per_maturity = 20U;
    constexpr float log_moneyness_slope = 0.2f;
    GeneratedRows rows = maturity_dependent_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, maturity_count),
        strikes_per_maturity,
        log_moneyness_slope
    );
    for (auto& row : rows.rows) {
        const float maturity = row.at("maturity").get<float>();
        row["moneyness"] = row.at("strike");
        row["reset_time"] = 0.5f * maturity;
        row.erase("strike");
    }
    rows.construction["grid"]["maturity"]["minimum"] = "1 / 12";
    rows.construction["grid"]["moneyness"] =
        rows.construction["grid"].at("strike");
    rows.construction["grid"].erase("strike");
    rows.construction["reset_time"] = {
        {"rule", "maturity / 2"},
    };

    write_product_dataset(
        "forward_start_puts_01",
        "Forward-Start Puts",
        dataset_path,
        catalog_path,
        url,
        {
            {"moneyness", "Strike multiplier applied to the reset spot."},
            {"reset_time", "Time when the floating strike is fixed."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "max(moneyness * S_reset - S_T, 0)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
