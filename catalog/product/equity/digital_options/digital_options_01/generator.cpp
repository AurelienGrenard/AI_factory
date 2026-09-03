// Generate cash-or-nothing calls on the standard equity grid.
#include "tools/datasets/parameter_dataset.hpp"
#include "common/dataset_validation.hpp"

#include <filesystem>
#include <string>

// Generate the digital-call dataset and catalog entry.
int main() {
    using namespace ai_factory::workbench::datasets;

    const std::filesystem::path dataset_path =
        "datasets/product/equity/digital_options/digital_options_01.json";
    const std::filesystem::path catalog_path =
        "catalog/product/equity/digital_options/digital_options_01/dataset.yaml";
    const std::string url =
        "https://datasets.ai-factory.example/v1/product/digital_options/"
        "digital_options_01.json";

    GeneratedRows rows = core_stress_exponential_strike_grid(
        linear_business_day_grid(21U, 756U, 45U),
        20U,
        0.2f,
        linear_business_day_grid(5U, 1764U, 10U),
        10U,
        0.2f
    );
    for (auto& row : rows.rows) row["cash_payoff"] = 1.0f;
    rows.construction["cash_payoff"] = {
        {"value", 1.0},
        {"rule", "constant normalized cash amount"},
    };

    write_product_dataset(
        "digital_options_01",
        "Digital Options",
        dataset_path,
        catalog_path,
        url,
        {
            {"strike", "Trigger strike in normalized spot units."},
            {"maturity", "Maturity in business days."},
            {"cash_payoff", "Cash amount paid when the side-oriented trigger is met."},
        },
        {
            {"expression", "cash_payoff * 1{side * (S_T - K) > 0}, side = +1 call / -1 put"},
        },
        rows
    );
    validate_product_dataset_file(dataset_path);
}
