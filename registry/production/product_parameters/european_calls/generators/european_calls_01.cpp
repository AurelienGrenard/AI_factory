// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/registry/dataset.hpp"

#include <filesystem>

// Generate the European-call parameter JSON and YAML databases.
int main() {
    using namespace ai_factory::workbench::registry;

    const std::filesystem::path json_path =
        "registry/production/product_parameters/european_calls/data/european_calls_01.json";
    const std::filesystem::path generator_path =
        "registry/production/product_parameters/european_calls/generators/european_calls_01.cpp";

    constexpr std::size_t maturity_count = 50U;
    constexpr std::size_t strikes_per_maturity = 20U;
    constexpr float log_moneyness_slope = 0.2f;
    GeneratedRows rows = maturity_dependent_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, maturity_count),
        strikes_per_maturity,
        log_moneyness_slope
    );
    // Preserve the exact lower maturity bound in human-readable metadata.
    rows.construction["grid"]["maturity"]["minimum"] = "1 / 12";

    write_product_database(
        "european_calls_01",
        "European Calls",
        json_path,
        generator_path,
        {
            {"strike", "Strike in normalized spot units."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "max(S_T - K, 0)"},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
}
