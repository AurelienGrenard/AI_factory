// Generate 20 maturity-dependent exponential strikes at each of 50 maturities.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cmath>
#include <filesystem>
#include <string>

// Generate the Down-and-in-option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/down_and_in_option/down_and_in_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/down_and_in_option/down_and_in_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/down_and_in_options/"
        "down_and_in_options_01.json";

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
        row["barrier"] = fminf(
            0.95f, strike * expf(-0.15f * sqrtf(maturity_years / 3.0f))
        );
    }

    write_product_dataset(
        "down_and_in_options_01",
        "Down-and-In Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"barrier", "Lower knock-in level monitored on the simulation grid."},
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "max(side * (S_T - K), 0) if min(S_[0,T]) <= barrier; side = +1 call / -1 put"},
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
