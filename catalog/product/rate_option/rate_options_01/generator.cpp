// Generate normalized rate_options with central strikes and sparse tail cases.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <filesystem>
#include <string>
#include <utility>

// Generate the forward rate option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/rate_option/rate_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/rate_option/rate_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/rate_options/"
        "rate_options_01.json";

    std::vector<float> core_strikes;
    core_strikes.reserve(25U);
    for (const float coordinate : linear_grid(-1.0f, 1.0f, 25U)) {
        const float cube = coordinate * coordinate * coordinate;
        core_strikes.push_back(
            coordinate < 0.0f ? 0.04f + 0.04f * cube
                              : 0.04f + 0.06f * cube
        );
    }
    const auto regime = [](const std::vector<std::uint32_t>& fixing_times,
                           const std::vector<std::uint32_t>& accrual_periods,
                           const std::vector<float>& strikes,
                           const std::string& description) {
        GeneratedRows generated;
        for (const std::uint32_t fixing_time_days : fixing_times) {
            for (const std::uint32_t accrual_period_days : accrual_periods) {
                for (const float strike : strikes) {
                    generated.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"fixing_time", fixing_time_days},
                    {
                        "payment_time",
                        fixing_time_days + accrual_period_days
                    },
                    {"accrual_period", accrual_period_days},
                    });
                }
            }
        }
        generated.construction = {
            {"method", "Cartesian grid"},
            {"rule", "Every fixing, accrual period, and strike combination."},
            {"description", description},
            {"fixing_times", fixing_times},
            {"accrual_periods", accrual_periods},
            {"strikes", strikes},
            {"notional", 1.0},
        };
        return generated;
    };
    GeneratedRows core = regime(
        linear_business_day_grid(63U, 1260U, 18U),
        {63U, 126U},
        core_strikes,
        "Representative fixing dates and strikes concentrated around 4%."
    );
    GeneratedRows stress = regime(
        linear_business_day_grid(5U, 3780U, 10U),
        {5U, 504U},
        {-0.10f, -0.02f, 0.04f, 0.15f, 0.35f},
        "Very short/long dates, short/long accruals, and negative/high strikes."
    );
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_product_dataset(
        "rate_options_01",
        "Forward Rate Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"notional", "Normalized contract notional."},
            {"strike", "Simple annualized forward rate option strike."},
            {"fixing_time", "Forward-rate fixing time T1 in business days."},
            {"payment_time", "Cashflow payment time T2 in business days."},
            {"accrual_period", "Business days in the interval [T1, T2]."},
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
