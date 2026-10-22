// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/parameter_dataset.hpp"
#include "datasets/validation.hpp"

#include <filesystem>
#include <string>

// Generate the Geometric-Asian-option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/geometric_asian_option/geometric_asian_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/geometric_asian_option/geometric_asian_options_01/generation.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/geometric_asian_options/"
        "geometric_asian_options_01.json";

    constexpr float log_moneyness_slope = 0.15f;
    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_business_day_grid(21U, 756U, 45U),
        20U,
        log_moneyness_slope,
        linear_business_day_grid(5U, 1260U, 10U),
        10U,
        log_moneyness_slope
    );
    // Preserve the exact lower maturity bound in human-readable metadata.

    write_spot_relative_equity_product_dataset(
        "geometric_asian_options_01",
        "Geometric Asian Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "max(side * (geometric_mean(S_[0,T]) - K), 0), side = +1 call / -1 put"},
            {"monitoring", "Simulation-grid endpoints including 0 and T."},
            {"payoff_homogeneity", "Scaling the entire spot path and all strike/barrier levels by c scales the payoff by c."},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
