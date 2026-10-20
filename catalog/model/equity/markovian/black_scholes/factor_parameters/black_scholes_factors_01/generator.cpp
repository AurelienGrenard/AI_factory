#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/black_scholes/parameters/black_scholes_01.json",
        "black_scholes_factors_01",
        "datasets/model/equity/markovian/black_scholes/factor_parameters/black_scholes_factors_01.json",
        "catalog/model/equity/markovian/black_scholes/factor_parameters/black_scholes_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/black_scholes/factor_parameters/black_scholes_factors_01.json");
}
