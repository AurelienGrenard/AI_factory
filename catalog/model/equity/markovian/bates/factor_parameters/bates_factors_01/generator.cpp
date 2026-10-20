#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/bates/parameters/bates_01.json",
        "bates_factors_01",
        "datasets/model/equity/markovian/bates/factor_parameters/bates_factors_01.json",
        "catalog/model/equity/markovian/bates/factor_parameters/bates_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/bates/factor_parameters/bates_factors_01.json");
}
