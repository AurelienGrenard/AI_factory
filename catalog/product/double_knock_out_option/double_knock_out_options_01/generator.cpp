// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cmath>
#include <filesystem>
#include <string>

// Generate the Double-knock-out-option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/double_knock_out_option/double_knock_out_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/double_knock_out_option/double_knock_out_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/double_knock_out_options/"
        "double_knock_out_options_01.json";

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
        const float strike = row.at("strike").get<float>();
        const std::uint32_t maturity_days =
            row.at("maturity").get<std::uint32_t>();
        const float maturity_years = business_days_to_years(maturity_days);
        const float width = 0.20f * sqrtf(maturity_years / 3.0f);
        row["lower_barrier"] = fminf(0.90f, strike * expf(-width));
        row["upper_barrier"] = fmaxf(1.10f, strike * expf(width));
    }
    // Preserve the exact lower maturity bound in human-readable metadata.

    write_product_dataset(
        "double_knock_out_options_01",
        "Double-Knock-Out Calls",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"lower_barrier", "Lower knock-out level."},
            {"upper_barrier", "Upper knock-out level."},
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "max(side * (S_T - K), 0) while S remains between both barriers; side = +1 call / -1 put"},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
