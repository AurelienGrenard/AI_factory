// Generate regular co-terminal physical-settlement Bermudan swaptions.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

using ai_factory::workbench::datasets::GeneratedRows;

struct ScheduleSpec {
    std::uint32_t payment_interval_days;
    std::uint32_t payment_count;
    std::uint32_t exercise_count;
    float accrual_fraction;
};

GeneratedRows regular_regime(
    const std::vector<std::uint32_t>& first_exercise_times,
    const std::vector<ScheduleSpec>& schedules,
    const std::vector<float>& strikes,
    const std::string& description
) {
    GeneratedRows generated;
    for (const std::uint32_t first_exercise_time_days : first_exercise_times) {
        for (const ScheduleSpec& schedule : schedules) {
            for (const float strike : strikes) {
                generated.rows.push_back({
                    {"notional", 1.0f},
                    {"strike", strike},
                    {"accrual_fraction", schedule.accrual_fraction},
                    {"first_exercise_time", first_exercise_time_days},
                    {"payment_interval", schedule.payment_interval_days},
                    {"payment_count", schedule.payment_count},
                    {"exercise_count", schedule.exercise_count},
                });
            }
        }
    }

    nlohmann::ordered_json schedule_rows = nlohmann::ordered_json::array();
    for (const ScheduleSpec& schedule : schedules) {
        schedule_rows.push_back({
            {"payment_interval", schedule.payment_interval_days},
            {"payment_count", schedule.payment_count},
            {"exercise_count", schedule.exercise_count},
            {"accrual_fraction", schedule.accrual_fraction},
        });
    }
    generated.construction = {
        {"method", "Cartesian grid"},
        {
            "rule",
            "Every first exercise time, co-terminal regular schedule, and "
            "strike combination."
        },
        {"description", description},
        {"first_exercise_times", first_exercise_times},
        {"schedules", schedule_rows},
        {"strikes", strikes},
        {"notional", 1.0},
    };
    return generated;
}

}  // namespace

int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/bermudan_swaption/"
        "bermudan_swaptions_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/bermudan_swaption/"
        "bermudan_swaptions_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/"
        "bermudan_swaptions/bermudan_swaptions_01.json";

    const GeneratedRows core = regular_regime(
        {
            126U, 252U, 504U, 756U, 1260U,
            1764U, 2520U, 3780U, 5040U, 7560U,
        },
        {
            {252U, 2U, 2U, 1.0f},
            {252U, 3U, 3U, 1.0f},
            {252U, 5U, 5U, 1.0f},
            {252U, 7U, 5U, 1.0f},
            {252U, 10U, 5U, 1.0f},
            {252U, 10U, 10U, 1.0f},
            {126U, 4U, 4U, 0.5f},
            {126U, 10U, 6U, 0.5f},
            {126U, 20U, 10U, 0.5f},
            {126U, 40U, 10U, 0.5f},
        },
        {
            0.005f, 0.0125f, 0.020f, 0.0275f, 0.035f,
            0.0425f, 0.050f, 0.065f, 0.080f,
        },
        "Six-month to thirty-year first exercise dates; two-to-twenty-year "
        "co-terminal swaps; annual or semiannual fixed legs; and two-to-ten "
        "exercise opportunities at ordinary fixed rates."
    );
    const GeneratedRows stress = regular_regime(
        {1U, 5U, 21U, 3780U, 7560U},
        {
            {1U, 2U, 2U, 1.0f / 252.0f},
            {5U, 12U, 4U, 5.0f / 252.0f},
            {21U, 120U, 12U, 1.0f / 12.0f},
            {63U, 200U, 8U, 0.25f},
            {252U, 100U, 10U, 1.0f},
        },
        {0.0f, 0.001f, 0.15f, 0.35f},
        "One-day through thirty-year first exercises, daily through annual "
        "fixed legs, two-payment through hundred-year co-terminal swaps, "
        "and zero or unusually high fixed rates."
    );
    const GeneratedRows rows = core_stress_rows(core, stress);

    write_product_dataset(
        "bermudan_swaptions_01",
        "Bermudan Swaptions",
        dataset_path,
        catalog_path,
        url,
        {
            {"notional", "Contract notional N."},
            {"strike", "Fixed swap rate K."},
            {
                "accrual_fraction",
                "Constant contractual coupon year fraction delta."
            },
            {
                "first_exercise_time",
                "First exercise and initial swap-start date in business days."
            },
            {
                "payment_interval",
                "Business days between payments and exercise opportunities."
            },
            {"payment_count", "Payments from first exercise to final maturity."},
            {"exercise_count", "Co-terminal exercise opportunities."},
        },
        {
            {
                "expression",
                "max over exercise policy of N * max(omega * (1 - "
                "P(t,T_N) - K * sum_{i>j} delta * P(t,T_i)), 0), "
                "omega = +1 payer / -1 receiver"
            },
            {
                "exercise_dates",
                "first_exercise_time + j * payment_interval, "
                "j = 0,...,exercise_count-1"
            },
            {"underlying", "At exercise j, the remaining co-terminal swap."},
            {"settlement", "physical"},
            {"normalization", "N = 1"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
