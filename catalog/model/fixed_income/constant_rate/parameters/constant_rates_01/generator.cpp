#include "tools/datasets/parameter_dataset.hpp"
#include "datasets/validation.hpp"

int main() {
    using namespace ai_factory::workbench::datasets;
    const auto rows = aligned_grid({
        {"rate", {-0.03f, -0.01f, 0.0f, 0.01f, 0.03f, 0.05f, 0.08f}}
    });
    constexpr auto dataset =
        "datasets/model/fixed_income/constant_rate/parameters/constant_rates_01.json";
    write_model_dataset(
        "constant_rates_01", "Constant short rate", dataset,
        "catalog/model/fixed_income/constant_rate/parameters/constant_rates_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/fixed_income/constant_rate/parameters/constant_rates_01.json",
        {{"rate", "Continuously compounded short rate."}},
        {{"short_rate", "r(t) = rate"}},
        rows);
    validate_model_dataset_file(dataset);
}
