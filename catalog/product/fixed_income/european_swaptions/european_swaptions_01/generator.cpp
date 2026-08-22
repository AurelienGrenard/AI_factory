// Generate regular physical-settlement European swaptions.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

using ai_factory::workbench::datasets::GeneratedRows;

struct ScheduleSpec {
    std::uint32_t payment_interval;
    std::uint32_t payment_count;
    float accrual_fraction;
};

// Expand a traceable regular-schedule grid in deterministic row order.
GeneratedRows regular_regime(
    const std::vector<std::uint32_t>& exercise_times,
    const std::vector<ScheduleSpec>& schedules,
    const std::vector<float>& strikes,
    const std::string& description
) {
    GeneratedRows generated;
    for (const std::uint32_t exercise_time : exercise_times) {
        for (const ScheduleSpec& schedule : schedules) {
            for (const float strike : strikes) {
                generated.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"exercise_time", exercise_time},
                    {"payment_interval", schedule.payment_interval},
                    {"payment_count", schedule.payment_count},
                    {"accrual_fraction", schedule.accrual_fraction},
                });
            }
        }
    }

    nlohmann::ordered_json schedule_rows = nlohmann::ordered_json::array();
    for (const ScheduleSpec& schedule : schedules) {
        schedule_rows.push_back({
            {"payment_interval", schedule.payment_interval},
            {"payment_count", schedule.payment_count},
            {"accrual_fraction", schedule.accrual_fraction},
        });
    }
    generated.construction = {
        {"method", "Cartesian grid"},
        {
            "rule",
            "Every exercise time, regular fixed-leg schedule, and strike "
            "combination."
        },
        {"description", description},
        {"exercise_times", exercise_times},
        {"schedules", schedule_rows},
        {"strikes", strikes},
        {"notional", 1.0},
    };
    return generated;
}

}  // namespace

// Generate the product dataset and its compact catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/fixed_income/european_swaptions/"
        "european_swaptions_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/fixed_income/european_swaptions/"
        "european_swaptions_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/"
        "european_swaptions/european_swaptions_01.json";

    std::vector<ScheduleSpec> core_schedules;
    core_schedules.reserve(10U);
    for (const std::uint32_t tenor_years : {1U, 2U, 5U, 10U, 20U}) {
        core_schedules.push_back({252U, tenor_years, 1.0f});
        core_schedules.push_back({126U, 2U * tenor_years, 0.5f});
    }

    GeneratedRows core = regular_regime(
        {63U, 126U, 252U, 504U, 756U,
         1260U, 1764U, 2520U, 3780U, 5040U},
        core_schedules,
        {0.005f, 0.0125f, 0.020f, 0.0275f, 0.035f,
         0.0425f, 0.050f, 0.065f, 0.080f},
        "Standard expiries, one-to-twenty-year swaps, annual or semiannual "
        "fixed legs, and strikes concentrated around ordinary rate levels."
    );
    GeneratedRows stress = regular_regime(
        {1U, 5U, 21U, 7560U, 12600U},
        {
            {5U, 1U, 1.0f / 52.0f},
            {21U, 12U, 1.0f / 12.0f},
            {21U, 600U, 1.0f / 12.0f},
            {63U, 200U, 0.25f},
            {252U, 50U, 1.0f},
        },
        {0.0f, 0.001f, 0.15f, 0.35f},
        "Very short or long expiries, one-payment through fifty-year fixed "
        "legs, monthly long schedules, and zero or unusually high strikes."
    );
    const GeneratedRows rows = core_stress_rows(
        std::move(core), std::move(stress)
    );

    write_product_dataset(
        "european_swaptions_01",
        "European Swaptions",
        dataset_path,
        catalog_path,
        url,
        {
            {"notional", "Contract notional N."},
            {"strike", "Fixed swap rate K."},
            {
                "exercise_time",
                "Exercise date and underlying swap start T_e in business days."
            },
            {
                "payment_interval",
                "Constant number of business days between fixed-leg payments."
            },
            {"payment_count", "Number of fixed-leg payments."},
            {
                "accrual_fraction",
                "Constant contractual year fraction delta for each coupon."
            },
        },
        {
            {
                "expression",
                "N * max(omega * (1 - P(T_e,T_N) - K * sum_i "
                "delta * P(T_e,T_i)), 0), omega = +1 payer / -1 receiver"
            },
            {"exercise_time", "T_e equals the underlying swap start."},
            {"settlement", "physical"},
            {"normalization", "N = 1"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
