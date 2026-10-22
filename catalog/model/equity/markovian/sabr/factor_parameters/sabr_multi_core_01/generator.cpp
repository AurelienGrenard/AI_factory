#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/model/equity/markovian/sabr/parameters/sabr_01.json",
        "sabr_multi_core_01",
        "datasets/model/equity/markovian/sabr/factor_parameters/sabr_multi_core_01.json",
        "catalog/model/equity/markovian/sabr/factor_parameters/sabr_multi_core_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/factor_parameters/sabr_multi_core_01.json",
        900U, 100U, 2026102103ULL);
}
