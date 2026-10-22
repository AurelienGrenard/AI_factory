#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/model/equity/markovian/heston/parameters/heston_01.json",
        "heston_multi_core_01",
        "datasets/model/equity/markovian/heston/factor_parameters/heston_multi_core_01.json",
        "catalog/model/equity/markovian/heston/factor_parameters/heston_multi_core_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/heston/factor_parameters/heston_multi_core_01.json",
        900U, 100U, 2026102101ULL);
}
