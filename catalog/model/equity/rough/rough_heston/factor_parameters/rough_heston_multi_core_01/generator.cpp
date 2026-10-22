#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/model/equity/rough/rough_heston/parameters/rough_heston_01.json",
        "rough_heston_multi_core_01",
        "datasets/model/equity/rough/rough_heston/factor_parameters/rough_heston_multi_core_01.json",
        "catalog/model/equity/rough/rough_heston/factor_parameters/rough_heston_multi_core_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/rough/rough_heston/factor_parameters/rough_heston_multi_core_01.json",
        900U, 100U, 2026102104ULL);
}
