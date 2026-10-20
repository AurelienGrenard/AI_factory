#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/rough/log_modulated_rough_bergomi/parameters/log_modulated_rough_bergomi_01.json",
        "log_modulated_rough_bergomi_factors_01",
        "datasets/model/equity/rough/log_modulated_rough_bergomi/factor_parameters/log_modulated_rough_bergomi_factors_01.json",
        "catalog/model/equity/rough/log_modulated_rough_bergomi/factor_parameters/log_modulated_rough_bergomi_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/rough/log_modulated_rough_bergomi/factor_parameters/log_modulated_rough_bergomi_factors_01.json");
}
