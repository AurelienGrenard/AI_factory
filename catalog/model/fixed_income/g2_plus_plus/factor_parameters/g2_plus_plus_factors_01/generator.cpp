#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json",
        "g2_plus_plus_factors_01",
        "datasets/model/fixed_income/g2_plus_plus/factor_parameters/g2_plus_plus_factors_01.json",
        "catalog/model/fixed_income/g2_plus_plus/factor_parameters/g2_plus_plus_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/fixed_income/g2_plus_plus/factor_parameters/g2_plus_plus_factors_01.json");
}
