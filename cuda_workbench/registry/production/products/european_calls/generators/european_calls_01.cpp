// Generate the local European-call database as a 20 x 50 Cartesian grid.
#include "tools/registry/parameter_database.hpp"

#include <filesystem>

int main() {
    using namespace ai_factory::workbench::registry;

    const std::filesystem::path json_path =
        "cuda_workbench/registry/production/products/european_calls/data/european_calls_01.json";
    const std::filesystem::path generator_path =
        "cuda_workbench/registry/production/products/european_calls/generators/european_calls_01.cpp";

    const GeneratedRows rows = cartesian_grid({
        {"strike", linear_grid(0.70f, 1.30f, 20U), "linear"},
        {"maturity", linear_grid(1.0f / 12.0f, 3.0f, 50U), "linear"},
    });

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
