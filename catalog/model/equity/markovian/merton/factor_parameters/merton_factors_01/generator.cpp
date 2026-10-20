#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/merton/parameters/merton_01.json",
        "merton_factors_01",
        "datasets/model/equity/markovian/merton/factor_parameters/merton_factors_01.json",
        "catalog/model/equity/markovian/merton/factor_parameters/merton_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/merton/factor_parameters/merton_factors_01.json");
}
