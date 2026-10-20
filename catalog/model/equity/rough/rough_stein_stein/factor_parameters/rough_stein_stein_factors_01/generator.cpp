#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/rough/rough_stein_stein/parameters/rough_stein_stein_01.json",
        "rough_stein_stein_factors_01",
        "datasets/model/equity/rough/rough_stein_stein/factor_parameters/rough_stein_stein_factors_01.json",
        "catalog/model/equity/rough/rough_stein_stein/factor_parameters/rough_stein_stein_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/rough/rough_stein_stein/factor_parameters/rough_stein_stein_factors_01.json");
}
