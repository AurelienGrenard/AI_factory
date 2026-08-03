// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cmath>
#include <filesystem>
#include <string>

// Generate the Down-and-out-put dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/down_and_out_puts/down_and_out_puts_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/down_and_out_puts/down_and_out_puts_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/down_and_out_puts/"
        "down_and_out_puts_01.json";

    constexpr std::size_t maturity_count = 50U;
    constexpr std::size_t strikes_per_maturity = 20U;
    constexpr float log_moneyness_slope = 0.2f;
    GeneratedRows rows = maturity_dependent_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, maturity_count),
        strikes_per_maturity,
        log_moneyness_slope
    );
    for (auto& row : rows.rows) {
        const float strike = row.at("strike").get<float>();
        const float maturity = row.at("maturity").get<float>();
        row["barrier"] = fminf(
            0.95f, strike * expf(-0.15f * sqrtf(maturity / 3.0f))
        );
    }
    rows.construction["grid"]["maturity"]["minimum"] = "1 / 12";

    write_product_dataset(
        "down_and_out_puts_01",
        "Down-and-Out Puts",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"barrier", "Lower knock-out level monitored on the simulation grid."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "max(K - S_T, 0) if min(S_[0,T]) > barrier; otherwise 0"},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
