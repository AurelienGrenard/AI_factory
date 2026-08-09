// Generate non-negative gap puts on the standard equity grid.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cmath>
#include <filesystem>
#include <string>

// Generate the gap-put dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/gap_options/gap_put_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/gap_options/gap_put_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/gap_options/"
        "gap_put_options_01.json";

    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, 45U),
        20U,
        0.2f,
        linear_grid(1.0f / 52.0f, 7.0f, 10U),
        10U,
        0.2f
    );
    for (std::size_t index = 0U; index < rows.rows.size(); ++index) {
        const float trigger = rows.rows[index].at("strike").get<float>();
        const float maturity = rows.rows[index].at("maturity").get<float>();
        const float relative_gap = 0.05f * sqrtf(maturity / 3.0f);
        const float payoff_strike = trigger * expf(relative_gap);
        rows.rows[index] = {
            {"trigger_strike", trigger},
            {"payoff_strike", payoff_strike},
            {"maturity", maturity},
        };
    }
    rows.construction["grid"]["trigger_strike"] =
        rows.construction["grid"].at("strike");
    rows.construction["grid"].erase("strike");
    rows.construction["payoff_strike"] = {
        {"rule", "trigger_strike * exp(0.05 * sqrt(T / 3))"},
    };

    write_product_dataset(
        "gap_put_options_01",
        "Gap Puts",
        dataset_path,
        catalog_path,
        url,
        {
            {"trigger_strike", "Strike that activates the terminal payoff."},
            {"payoff_strike", "Strike from which the terminal spot is subtracted."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "(payoff_strike - S_T) * 1{S_T < trigger_strike}"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
