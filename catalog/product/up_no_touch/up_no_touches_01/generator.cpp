// Generate 20 maturity-dependent upper barriers at each of 50 maturities.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cmath>
#include <filesystem>
#include <string>

// Generate the Up-no-touch dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/up_no_touch/up_no_touches_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/up_no_touch/up_no_touches_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/up_no_touches/"
        "up_no_touches_01.json";

    constexpr float log_moneyness_slope = 0.2f;
    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_business_day_grid(21U, 756U, 45U),
        20U,
        log_moneyness_slope,
        linear_business_day_grid(5U, 1764U, 10U),
        10U,
        log_moneyness_slope
    );
    for (std::size_t index = 0U; index < rows.rows.size(); ++index) {
        auto& row = rows.rows[index];
        const bool stress = index >= 900U;
        const std::uint32_t maturity_days =
            row.at("maturity").get<std::uint32_t>();
        const float maturity_years = business_days_to_years(maturity_days);
        const float log_strike = logf(row.at("strike").get<float>());
        const float regime_slope = stress
            ? log_moneyness_slope
            : log_moneyness_slope;
        const float grid_position = (
            log_strike + regime_slope * maturity_years
        ) / (2.0f * regime_slope * maturity_years);
        row.erase("strike");
        row["barrier"] = stress
            ? expf(0.01f + grid_position
                * (0.80f + 0.40f * sqrtf(maturity_years / 7.0f)))
            : expf(0.05f + grid_position
                * (0.10f + 0.20f * sqrtf(maturity_years / 3.0f)));
        row["cash_payoff"] = 1.0f;
    }
    rows.construction["rule"] =
        "For each T, the upper barrier is linearly spaced in log-space.";
    rows.construction["grid"].erase("strike");
    rows.construction["grid"]["barrier"] = {
        {"spacing", "linear in log-barrier"},
        {"core", {
            {"count_per_maturity", 20},
            {"conditional_bounds", "[exp(0.05), exp(0.15 + 0.20 sqrt((T/252) / 3))]"},
        }},
        {"stress", {
            {"count_per_maturity", 10},
            {"conditional_bounds", "[exp(0.01), exp(0.81 + 0.40 sqrt((T/252) / 7))]"},
        }},
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
            {"maturity", "Maturity in business days."},
        },
        {
            {"expression", "cash_payoff if max(S_[0,T]) < barrier; otherwise 0"},
            {"payment_time", "maturity"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
