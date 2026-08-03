// Generate non-negative gap calls on the standard equity grid.
#include "tools/datasets/dataset.hpp"
#include "tools/datasets/dataset_validation.hpp"

#include <cmath>
#include <filesystem>
#include <string>

// Generate the gap-call dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/gap_calls/gap_calls_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/gap_calls/gap_calls_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/gap_calls/"
        "gap_calls_01.json";

    constexpr std::size_t maturity_count = 50U;
    constexpr std::size_t strikes_per_maturity = 20U;
    GeneratedRows rows = maturity_dependent_exponential_strike_grid(
        linear_grid(1.0f / 12.0f, 3.0f, maturity_count),
        strikes_per_maturity,
        0.2f
    );
    for (std::size_t index = 0U; index < rows.rows.size(); ++index) {
        const float trigger = rows.rows[index].at("strike").get<float>();
        const float maturity = rows.rows[index].at("maturity").get<float>();
        const float relative_gap = 0.05f * sqrtf(maturity / 3.0f);
        const float payoff_strike = trigger * expf(-relative_gap);
        rows.rows[index] = {
            {"trigger_strike", trigger},
            {"payoff_strike", payoff_strike},
            {"maturity", maturity},
        };
    }
    rows.construction["grid"]["maturity"]["minimum"] = "1 / 12";
    rows.construction["grid"]["trigger_strike"] =
        rows.construction["grid"].at("strike");
    rows.construction["grid"].erase("strike");
    rows.construction["payoff_strike"] = {
        {"rule", "trigger_strike * exp(-0.05 * sqrt(T / 3))"},
    };

    write_product_dataset(
        "gap_calls_01",
        "Gap Calls",
        dataset_path,
        catalog_path,
        url,
        {
            {"trigger_strike", "Strike that activates the terminal payoff."},
            {"payoff_strike", "Strike subtracted from the terminal spot."},
            {"maturity", "Maturity in years."},
        },
        {
            {"expression", "(S_T - payoff_strike) * 1{S_T > trigger_strike}"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
