#include "common/dataset_validation.hpp"
#include "tools/datasets/parameter_dataset.hpp"

int main() {
    using namespace ai_factory::workbench::datasets;
    GeneratedRows rows;
    rows.rows.push_back({
        {"strike", 0.95f},
        {"maturity", 21U}
    });
    rows.construction = {
        {"method", "fixed reference contract"},
        {"purpose", "multi-asset pricing integration"}
    };
    write_product_dataset(
        "worst_of_calls_01", "Worst-of calls",
        "datasets/product/worst_of_call/worst_of_calls_01.json",
        "catalog/product/worst_of_call/worst_of_calls_01/generation.yaml",
        "https://datasets.ai-factory.example/v1/product/worst_of_call/worst_of_calls_01.json",
        {
            {"strike", "Strike on the minimum terminal spot divided by its own initial spot."},
            {"maturity", "Maturity in business days."}
        },
        {
            {"expression", "max(min_i(S_i(T)/S_i(0)) - strike, 0)"},
            {"notional", "one"}
        },
        rows,
        {
            {"strike", {{"mode", "strike_on_minimum_normalized_spot"},
                        {"unit", "dimensionless"},
                        {"underlying", "min_i(S_i(T)/S_i(0))"}}}
        }
    );
    validate_product_dataset_file(
        "datasets/product/worst_of_call/worst_of_calls_01.json"
    );
}
