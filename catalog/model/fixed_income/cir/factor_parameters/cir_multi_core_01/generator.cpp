#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/model/fixed_income/cir/parameters/cir_01.json",
        "cir_multi_core_01",
        "datasets/model/fixed_income/cir/factor_parameters/cir_multi_core_01.json",
        "catalog/model/fixed_income/cir/factor_parameters/cir_multi_core_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/fixed_income/cir/factor_parameters/cir_multi_core_01.json",
        900U, 100U, 2026102106ULL);
}
