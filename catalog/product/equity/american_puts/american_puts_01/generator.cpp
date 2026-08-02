// Generate the put grid and a feasible random Bermudan exercise convention.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the American-put dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/american_puts/american_puts_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/american_puts/american_puts_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/american_puts/"
        "american_puts_01.json";

    constexpr std::size_t maturity_count = 50U;
    constexpr std::size_t strikes_per_maturity = 20U;
    constexpr float log_moneyness_slope = 0.2f;
    constexpr std::size_t minimum_exercise_dates = 2U;
    constexpr std::uint64_t exercise_interval_seed = 731000101ULL;

    // Reuse exactly the European-call strike and maturity construction.
    GeneratedRows rows = maturity_dependent_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, maturity_count),
        strikes_per_maturity,
        log_moneyness_slope
    );
    rows.construction["grid"]["maturity"]["minimum"] = "1 / 12";

    // Select monthly or semi-monthly exercise when maturity permits it.
    assign_uniform_exercise_intervals(
        rows,
        {
            {1.0f / 12.0f, "1 / 12"},
            {1.0f / 24.0f, "1 / 24"},
        },
        minimum_exercise_dates,
        exercise_interval_seed
    );

    write_product_dataset(
        "american_puts_01",
        "American Puts",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Strike in normalized spot units."},
            {"maturity", "Maturity in years."},
            {"exercise_interval", "Time in years between exercise dates."},
        },
        {
            {"expression", "max(K - S_t, 0) at each exercise date"},
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
