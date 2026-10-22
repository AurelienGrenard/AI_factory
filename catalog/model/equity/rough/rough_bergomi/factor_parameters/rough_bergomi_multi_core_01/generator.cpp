#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json",
        "rough_bergomi_multi_core_01",
        "datasets/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_multi_core_01.json",
        "catalog/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_multi_core_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_multi_core_01.json",
        900U, 100U, 2026102105ULL);
}
