#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json",
        "rough_heston_factors_01",
        "datasets/model/equity/rough/rough_heston/factor_parameters/rough_heston_factors_01.json",
        "catalog/model/equity/rough/rough_heston/factor_parameters/rough_heston_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/rough/rough_heston/factor_parameters/rough_heston_factors_01.json");
}
