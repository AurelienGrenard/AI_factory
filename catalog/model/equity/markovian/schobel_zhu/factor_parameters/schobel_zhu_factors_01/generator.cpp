#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_factorized_model_dataset(
        "datasets/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json",
        "schobel_zhu_factors_01",
        "datasets/model/equity/markovian/schobel_zhu/factor_parameters/schobel_zhu_factors_01.json",
        "catalog/model/equity/markovian/schobel_zhu/factor_parameters/schobel_zhu_factors_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/schobel_zhu/factor_parameters/schobel_zhu_factors_01.json");
}
