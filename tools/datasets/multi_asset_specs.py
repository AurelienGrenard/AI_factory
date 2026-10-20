"""Selected product, factor and rate datasets for multi-asset composition."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceDatasetSpec:
    cmake_target: str
    dataset_kind: str
    generator_path: str
    dataset_path: str

    @property
    def recipe_yaml_path(self) -> str:
        return str(Path(self.generator_path).with_name("recipe.yaml"))

    @property
    def generation_yaml_path(self) -> str:
        return str(Path(self.generator_path).with_name("generation.yaml"))


_FACTOR_MODELS = {
    "equity/markovian": (
        "bates", "black_scholes", "cev", "heston", "heston_3_2",
        "kou", "merton", "normal_inverse_gaussian", "sabr",
        "schobel_zhu", "stein_stein", "variance_gamma",
    ),
    "equity/rough": (
        "log_modulated_rough_bergomi", "quadratic_rough_heston",
        "rough_bergomi", "rough_heston", "rough_sabr",
        "rough_stein_stein",
    ),
    "fixed_income": ("g2", "g2_plus_plus"),
}

REFERENCE_DATASET_SPECS = (
    ReferenceDatasetSpec(
        "generate_worst_of_calls_01", "product_parameters",
        "catalog/product/worst_of_call/worst_of_calls_01/generator.cpp",
        "datasets/product/worst_of_call/worst_of_calls_01.json",
    ),
    *(
        ReferenceDatasetSpec(
            f"generate_{model}_factors_01", "model_parameters",
            f"catalog/model/{market}/{model}/factor_parameters/"
            f"{model}_factors_01/generator.cpp",
            f"datasets/model/{market}/{model}/factor_parameters/"
            f"{model}_factors_01.json",
        )
        for market, models in _FACTOR_MODELS.items()
        for model in models
    ),
    ReferenceDatasetSpec(
        "generate_constant_rates_01", "model_parameters",
        "catalog/model/fixed_income/constant_rate/parameters/"
        "constant_rates_01/generator.cpp",
        "datasets/model/fixed_income/constant_rate/parameters/"
        "constant_rates_01.json",
    ),
)
