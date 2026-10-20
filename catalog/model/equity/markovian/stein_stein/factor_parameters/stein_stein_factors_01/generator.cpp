#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/stein_stein/parameters/stein_stein_01.json",
        "stein_stein_factors_01",
        "datasets/model/equity/markovian/stein_stein/factor_parameters/stein_stein_factors_01.json",
        "catalog/model/equity/markovian/stein_stein/factor_parameters/stein_stein_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/stein_stein/factor_parameters/stein_stein_factors_01.json");
}
