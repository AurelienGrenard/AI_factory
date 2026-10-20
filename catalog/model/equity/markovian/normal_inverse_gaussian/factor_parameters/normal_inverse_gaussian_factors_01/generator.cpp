#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/normal_inverse_gaussian/parameters/normal_inverse_gaussian_01.json",
        "normal_inverse_gaussian_factors_01",
        "datasets/model/equity/markovian/normal_inverse_gaussian/factor_parameters/normal_inverse_gaussian_factors_01.json",
        "catalog/model/equity/markovian/normal_inverse_gaussian/factor_parameters/normal_inverse_gaussian_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/normal_inverse_gaussian/factor_parameters/normal_inverse_gaussian_factors_01.json");
}
