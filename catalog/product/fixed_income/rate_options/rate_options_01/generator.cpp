// Generate normalized rate_options with central strikes and sparse tail cases.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <cmath>
#include <string>
#include <utility>

namespace {

std::vector<float> actual_360_grid(
    int first_day,
    int last_day,
    std::size_t point_count
) {
    std::vector<float> times;
    times.reserve(point_count);
    for (std::size_t index = 0U; index < point_count; ++index) {
        const double weight = static_cast<double>(index)
            / static_cast<double>(point_count - 1U);
        const int day = static_cast<int>(std::lround(
            first_day + weight * (last_day - first_day)
        ));
        times.push_back(static_cast<float>(day) / 360.0f);
    }
    return times;
}

}  // namespace

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

    std::vector<float> core_strikes;
    core_strikes.reserve(25U);
    for (const float coordinate : linear_grid(-1.0f, 1.0f, 25U)) {
        const float cube = coordinate * coordinate * coordinate;
        core_strikes.push_back(
            coordinate < 0.0f ? 0.04f + 0.04f * cube
                              : 0.04f + 0.06f * cube
        );
    }
    const auto regime = [](const std::vector<float>& fixing_times,
                           const std::vector<float>& accrual_periods,
                           const std::vector<float>& strikes,
                           const std::string& description) {
        GeneratedRows generated;
        for (const float fixing_time : fixing_times) {
            for (const float accrual_period : accrual_periods) {
                for (const float strike : strikes) {
                    generated.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"fixing_time", fixing_time},
                    {"payment_time", fixing_time + accrual_period},
                    {"accrual_period", accrual_period},
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
        actual_360_grid(90, 1800, 18U),
        {0.25f, 0.5f},
        core_strikes,
        "Representative fixing dates and strikes concentrated around 4%."
    );
    GeneratedRows stress = regime(
        actual_360_grid(7, 5400, 10U),
        {7.0f / 360.0f, 2.0f},
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
