#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/fixed_income/g2/parameters/g2_01.json",
        "g2_factors_01",
        "datasets/model/fixed_income/g2/factor_parameters/g2_factors_01.json",
        "catalog/model/fixed_income/g2/factor_parameters/g2_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/fixed_income/g2/factor_parameters/g2_factors_01.json");
}
