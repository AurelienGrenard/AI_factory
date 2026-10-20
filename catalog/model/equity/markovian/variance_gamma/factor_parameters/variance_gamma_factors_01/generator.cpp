#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json",
        "variance_gamma_factors_01",
        "datasets/model/equity/markovian/variance_gamma/factor_parameters/variance_gamma_factors_01.json",
        "catalog/model/equity/markovian/variance_gamma/factor_parameters/variance_gamma_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/variance_gamma/factor_parameters/variance_gamma_factors_01.json");
}
