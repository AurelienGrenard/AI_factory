// Generate the call grid and a feasible discrete exercise convention.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the American-option dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/american_options/american_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/american_options/american_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/american_options/"
        "american_options_01.json";

    constexpr float log_moneyness_slope = 0.2f;
    constexpr std::size_t minimum_exercise_dates = 2U;
    constexpr std::uint64_t exercise_interval_seed = 731000101ULL;

    // Reuse exactly the European-call strike and maturity construction.
    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, 45U),
        20U,
        log_moneyness_slope,
        linear_grid(1.0f / 26.0f, 7.0f, 10U),
        10U,
        log_moneyness_slope
    );

    // Select monthly or semi-monthly exercise when maturity permits it.
    assign_uniform_exercise_intervals(
        rows,
        {
            {1.0f / 12.0f, "1 / 12"},
            {1.0f / 24.0f, "1 / 24"},
            {1.0f / 52.0f, "1 / 52"},
        },
        minimum_exercise_dates,
        exercise_interval_seed
    );

    write_product_dataset(
        "american_options_01",
        "American Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"maturity", "Maturity in years."},
            {"exercise_interval", "Time in years between exercise dates."},
        },
        {
            {"expression", "max(side * (S_t - K), 0) at each exercise date; side = +1 call / -1 put"},
            {
                "interpretation",
                "Continuous American exercise is approximated on discrete dates."
            },
            {"scaling_rule", "V(s, K) = s * V(1, K / s)"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
