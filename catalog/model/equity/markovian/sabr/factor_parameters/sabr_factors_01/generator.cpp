#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/sabr/parameters/sabr_01.json",
        "sabr_factors_01",
        "datasets/model/equity/markovian/sabr/factor_parameters/sabr_factors_01.json",
        "catalog/model/equity/markovian/sabr/factor_parameters/sabr_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/factor_parameters/sabr_factors_01.json");
}
