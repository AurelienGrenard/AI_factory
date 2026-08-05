// Generate normalized rate_options with central strikes and sparse tail cases.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the forward rate option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/fixed_income/rate_options/rate_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/fixed_income/rate_options/rate_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/rate_options/"
        "rate_options_01.json";

    constexpr std::size_t fixing_count = 20U;
    constexpr std::size_t strike_count = 25U;
    const std::vector<float> fixing_times =
        linear_grid(0.25f, 5.0f, fixing_count);
    const std::vector<float> accrual_periods = {0.25f, 0.5f};

    // A cubic map concentrates strikes near 4% and keeps 0%/10% in the tails.
    std::vector<float> strikes;
    strikes.reserve(strike_count);
    for (const float coordinate : linear_grid(-1.0f, 1.0f, strike_count)) {
        const float cube = coordinate * coordinate * coordinate;
        strikes.push_back(
            coordinate < 0.0f ? 0.04f + 0.04f * cube
                              : 0.04f + 0.06f * cube
        );
    }

    GeneratedRows rows;
    rows.rows.reserve(
        fixing_times.size() * accrual_periods.size() * strikes.size()
    );
    for (const float fixing_time : fixing_times) {
        for (const float accrual_period : accrual_periods) {
            for (const float strike : strikes) {
                rows.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"fixing_time", fixing_time},
                    {"payment_time", fixing_time + accrual_period},
                    {"accrual_period", accrual_period},
                });
            }
        }
    }
    rows.construction = {
        {"method", "Cartesian grid"},
        {"rule", "Every fixing, accrual period, and strike combination."},
        {"grid", {
            {"fixing_time", {
                {"minimum", "1 / 4"}, {"maximum", 5.0f},
                {"count", fixing_count}, {"spacing", "linear"},
            }},
            {"accrual_period", {
                {"values", {"1 / 4", "1 / 2"}},
                {"count", accrual_periods.size()},
            }},
            {"strike", {
                {"minimum", "0 %"}, {"central_value", "4 %"},
                {"maximum", "10 %"}, {"count", strike_count},
                {"spacing", "cubic concentration around 4 %"},
            }},
            {"notional", {{"value", 1.0f}}},
        }},
    };

    write_product_dataset(
        "rate_options_01",
        "Forward Rate Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"notional", "Normalized contract notional."},
            {"strike", "Simple annualized forward rate option strike."},
            {"fixing_time", "Forward-rate fixing time T1 in years."},
            {"payment_time", "Cashflow payment time T2 in years."},
            {"accrual_period", "Year fraction delta for [T1, T2]."},
        },
        {
            {"expression", "N * delta * max(side * (L(T1; T1, T2) - K), 0), side = +1 caplet / -1 floorlet"},
            {"payment_time", "T2"},
            {"normalization", "N = 1"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
