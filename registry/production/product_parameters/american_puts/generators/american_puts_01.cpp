// Generate the put grid and a feasible random Bermudan exercise convention.
#include "tools/registry/dataset.hpp"

#include <filesystem>

// Generate the American-put parameter JSON and YAML databases.
int main() {
    using namespace ai_factory::workbench::registry;

    const std::filesystem::path json_path =
        "registry/production/product_parameters/american_puts/data/american_puts_01.json";
    const std::filesystem::path generator_path =
        "registry/production/product_parameters/american_puts/generators/american_puts_01.cpp";

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

    write_product_database(
        "american_puts_01",
        "American Puts",
        json_path,
        generator_path,
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
}
