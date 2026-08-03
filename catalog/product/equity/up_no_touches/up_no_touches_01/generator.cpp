// Generate 20 maturity-dependent upper barriers at each of 50 maturities.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cmath>
#include <filesystem>
#include <string>

// Generate the Up-no-touch dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/up_no_touches/up_no_touches_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/up_no_touches/up_no_touches_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/up_no_touches/"
        "up_no_touches_01.json";

    constexpr std::size_t maturity_count = 50U;
    constexpr std::size_t barriers_per_maturity = 20U;
    constexpr float log_moneyness_slope = 0.2f;
    GeneratedRows rows = maturity_dependent_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, maturity_count),
        barriers_per_maturity,
        log_moneyness_slope
    );
    for (auto& row : rows.rows) {
        const float maturity = row.at("maturity").get<float>();
        const float log_strike = logf(row.at("strike").get<float>());
        const float grid_position = (
            log_strike + log_moneyness_slope * maturity
        ) / (2.0f * log_moneyness_slope * maturity);
        row.erase("strike");
        row["barrier"] = expf(
            0.05f + grid_position
                * (0.10f + 0.20f * sqrtf(maturity / 3.0f))
        );
        row["cash_payoff"] = 1.0f;
    }
    rows.construction["grid"]["maturity"]["minimum"] = "1 / 12";
    rows.construction["rule"] =
        "For each T, the upper barrier is linearly spaced in log-space.";
    rows.construction["grid"].erase("strike");
    rows.construction["grid"]["barrier"] = {
        {"count_per_maturity", barriers_per_maturity},
        {"spacing", "linear in log-barrier"},
        {
            "conditional_bounds",
            "[exp(0.05), exp(0.15 + 0.20 sqrt(T / 3))]"
        },
    };
    rows.construction["cash_payoff"] = 1.0f;

    write_product_dataset(
        "up_no_touches_01",
        "Up No-Touches",
        dataset_path,
        catalog_path,
        url,
        {
            {"barrier", "Upper no-touch level in normalized spot units."},
            {"cash_payoff", "Cash amount paid at maturity without a touch."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "cash_payoff if max(S_[0,T]) < barrier; otherwise 0"},
            {"payment_time", "maturity"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
