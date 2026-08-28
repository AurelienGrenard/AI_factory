"""Typed repository capability matrix composed with the pricing manifest.

The C++ algorithms remain in ``src/``.  This module owns the published
compatibility matrix, dataset-recipe paths and CMake registration units,
including explicitly deferred capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass

from manifest import (
    AMERICAN_RECIPE_SPECS,
    BLACK_SCHOLES_CLOSED_FORM_PRODUCTS,
    MODEL_RECIPE_SPECS,
    PRICE_VARIANTS,
    ROUGH_N_FACTOR_MODELS,
    ROUGH_VOLTERRA_MODELS,
)


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EngineSpec:
    name: str
    asset_class: str
    algorithm: str
    optional_dependency: str | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display: str
    asset_class: str
    family: str
    parameter_dataset_id: str
    pricing_engine: str
    sample_engine: str | None
    sample_binding_status: str
    sample_recipe_status: str
    sample_requirement: str | None = None


@dataclass(frozen=True)
class ProductSpec:
    name: str
    asset_class: str
    catalog_folder: str
    parameter_dataset_ids: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    dataset_kind: str
    asset_class: str
    recipe_path: str
    generation: str
    status: str
    engine: str | None = None
    model: str | None = None
    curve: str | None = None
    product: str | None = None
    variant: str | None = None
    condition: str | None = None


@dataclass(frozen=True)
class CapabilityExceptionSpec:
    identifier: str
    scope: str
    owner: str
    rationale: str
    closure_condition: str


ENGINE_SPECS = (
    EngineSpec("equity_closed_form", "equity", "closed-form CUDA"),
    EngineSpec("equity_markovian", "equity", "Markovian CUDA Monte Carlo"),
    EngineSpec(
        "equity_volterra_fft",
        "equity",
        "Gaussian-Volterra hybrid FFT CUDA Monte Carlo",
        "mathDx/cuFFTDx",
    ),
    EngineSpec(
        "equity_n_factor",
        "equity",
        "prepared N-factor Markovian CUDA Monte Carlo",
    ),
    EngineSpec(
        "equity_lsm_fixed",
        "equity",
        "fixed-step Longstaff-Schwartz CUDA",
    ),
    EngineSpec(
        "equity_lsm_exact",
        "equity",
        "exact-transition Longstaff-Schwartz CUDA",
    ),
    EngineSpec(
        "fixed_income_closed_form",
        "fixed_income",
        "fixed-income closed-form CUDA",
    ),
    EngineSpec(
        "fixed_income_lsm",
        "fixed_income",
        "fixed-income Longstaff-Schwartz CUDA",
    ),
    EngineSpec("sample_markovian", "equity", "Markovian CUDA sampling"),
    EngineSpec("sample_n_factor", "equity", "N-factor CUDA sampling"),
    EngineSpec(
        "sample_volterra_fft",
        "equity",
        "Gaussian-Volterra hybrid FFT CUDA sampling",
        "mathDx/cuFFTDx",
    ),
    EngineSpec(
        "sample_fixed_income",
        "fixed_income",
        "fixed-income CUDA sampling",
    ),
)
ENGINE_BY_NAME = {engine.name: engine for engine in ENGINE_SPECS}


EQUITY_SAMPLE_MODELS = frozenset(
    model.name for model in MODEL_RECIPE_SPECS
)
ROUGH_VOLTERRA_NAMES = frozenset(
    model for model, _ in ROUGH_VOLTERRA_MODELS
)
ROUGH_N_FACTOR_NAMES = frozenset(
    model for model, _ in ROUGH_N_FACTOR_MODELS
)


def _equity_model_family(name: str) -> str:
    if name in ROUGH_VOLTERRA_NAMES:
        return "rough_volterra"
    if name in ROUGH_N_FACTOR_NAMES:
        return "rough_n_factor"
    return "markovian"


def _equity_pricing_engine(backend: str) -> str:
    return {
        "markovian": "equity_markovian",
        "volterra": "equity_volterra_fft",
        "n_factor": "equity_n_factor",
    }[backend]


def _equity_sample_engine(name: str) -> str:
    if name in ROUGH_VOLTERRA_NAMES:
        return "sample_volterra_fft"
    if name in ROUGH_N_FACTOR_NAMES:
        return "sample_n_factor"
    return "sample_markovian"


MODEL_SPECS = tuple(
    ModelSpec(
        model.name,
        model.display,
        "equity",
        _equity_model_family(model.name),
        f"{model.name}_01",
        _equity_pricing_engine(model.backend),
        _equity_sample_engine(model.name)
        if model.name in EQUITY_SAMPLE_MODELS else None,
        "available" if model.name in EQUITY_SAMPLE_MODELS else "deferred",
        "available",
        "AI_FACTORY_MATHDX_ROOT" if model.name in ROUGH_VOLTERRA_NAMES else None,
    )
    for model in MODEL_RECIPE_SPECS
) + tuple(
    ModelSpec(
        name,
        display,
        "fixed_income",
        "markovian",
        f"{name}_01",
        "fixed_income_closed_form",
        "sample_fixed_income",
        "available",
        "available",
    )
    for name, display in (
        ("cir", "CIR"),
        ("g2", "G2"),
        ("g2_plus_plus", "G2++"),
        ("hull_white", "Hull-White"),
        ("ornstein_uhlenbeck", "Ornstein-Uhlenbeck"),
        ("vasicek", "Vasicek"),
    )
)
MODEL_BY_NAME = {model.name: model for model in MODEL_SPECS}


def _equity_product_specs() -> tuple[ProductSpec, ...]:
    datasets: dict[tuple[str, str], set[str]] = {}
    for variant in PRICE_VARIANTS:
        key = (variant.product, variant.product_dataset_folder)
        datasets.setdefault(key, set()).add(variant.product_dataset_id)
    result = [
        ProductSpec(
            product,
            "equity",
            folder,
            tuple(sorted(dataset_ids)),
        )
        for (product, folder), dataset_ids in sorted(datasets.items())
    ]
    result.append(ProductSpec(
        "american_option",
        "equity",
        "american_options",
        ("american_options_01",),
    ))
    return tuple(sorted(result, key=lambda product: product.name))


PRODUCT_SPECS = _equity_product_specs() + (
    ProductSpec(
        "bermudan_swaption",
        "fixed_income",
        "bermudan_swaptions",
        ("bermudan_swaptions_01",),
    ),
    ProductSpec(
        "european_swaption",
        "fixed_income",
        "european_swaptions",
        ("european_swaptions_01",),
    ),
    ProductSpec(
        "rate_option",
        "fixed_income",
        "rate_options",
        ("rate_options_01",),
    ),
    ProductSpec(
        "zero_coupon_bond_option",
        "fixed_income",
        "zero_coupon_bond_options",
        ("zero_coupon_bond_options_01",),
    ),
)
PRODUCT_BY_NAME = {
    (product.asset_class, product.name): product
    for product in PRODUCT_SPECS
}


CURVE_DATASETS = (
    ("nelson_siegel", "nelson_siegel_01"),
    ("svensson", "svensson_01"),
)


FIXED_INCOME_VARIANTS = {
    "caplets": "rate_option",
    "floorlets": "rate_option",
    "zero_coupon_bond_calls": "zero_coupon_bond_option",
    "zero_coupon_bond_puts": "zero_coupon_bond_option",
    "european_payer_swaptions": "european_swaption",
    "european_receiver_swaptions": "european_swaption",
    "bermudan_payer_swaptions": "bermudan_swaption",
    "bermudan_receiver_swaptions": "bermudan_swaption",
}
FIXED_INCOME_ALL_VARIANTS = tuple(FIXED_INCOME_VARIANTS)
FIXED_INCOME_NO_EUROPEAN_VARIANTS = tuple(
    variant for variant in FIXED_INCOME_ALL_VARIANTS
    if not variant.startswith("european_")
)
FIXED_INCOME_CAPABILITIES = (
    ("cir", None, FIXED_INCOME_ALL_VARIANTS),
    ("g2", None, FIXED_INCOME_NO_EUROPEAN_VARIANTS),
    ("g2_plus_plus", "nelson_siegel", FIXED_INCOME_NO_EUROPEAN_VARIANTS),
    ("g2_plus_plus", "svensson", FIXED_INCOME_NO_EUROPEAN_VARIANTS),
    ("hull_white", "nelson_siegel", FIXED_INCOME_ALL_VARIANTS),
    ("hull_white", "svensson", FIXED_INCOME_ALL_VARIANTS),
    ("ornstein_uhlenbeck", None, FIXED_INCOME_ALL_VARIANTS),
    ("vasicek", None, FIXED_INCOME_ALL_VARIANTS),
)


def _model_parameter_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            model.parameter_dataset_id,
            "model_parameters",
            model.asset_class,
            f"catalog/model/{model.asset_class}/{model.name}/parameters/"
            f"{model.parameter_dataset_id}/generator.cpp",
            "generated",
            "available",
            model=model.name,
        )
        for model in MODEL_SPECS
    )


def _product_parameter_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset_id,
            "product_parameters",
            product.asset_class,
            f"catalog/product/{product.asset_class}/{product.catalog_folder}/"
            f"{dataset_id}/generator.cpp",
            "declared_manual",
            "available",
            product=product.name,
        )
        for product in PRODUCT_SPECS
        for dataset_id in product.parameter_dataset_ids
    )


def _curve_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset_id,
            "curve_parameters",
            "fixed_income",
            f"catalog/curve/{curve}/{dataset_id}/generator.cpp",
            "declared_manual",
            "available",
            curve=curve,
        )
        for curve, dataset_id in CURVE_DATASETS
    )


def _ordinary_equity_price_dataset_specs() -> tuple[DatasetSpec, ...]:
    result = []
    for model in MODEL_RECIPE_SPECS:
        for variant in PRICE_VARIANTS:
            dataset_id = f"{model.name}_01__{variant.name}_01__01"
            engine = _equity_pricing_engine(model.backend)
            if (
                model.name == "black_scholes"
                and variant.product in BLACK_SCHOLES_CLOSED_FORM_PRODUCTS
            ):
                engine = "equity_closed_form"
            result.append(DatasetSpec(
                dataset_id,
                "prices",
                "equity",
                f"catalog/model/equity/{model.name}/prices/{variant.name}/"
                f"{dataset_id}/generator.cpp",
                "generated",
                "available",
                engine,
                model.name,
                product=variant.product,
                variant=variant.name,
                condition=(
                    "AI_FACTORY_MATHDX_ROOT"
                    if engine == "equity_volterra_fft" else None
                ),
            ))
    return tuple(result)


def _american_price_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            f"{model.model}_01__american_{side}s_01__01",
            "prices",
            "equity",
            f"catalog/model/equity/{model.model}/prices/american_{side}s/"
            f"{model.model}_01__american_{side}s_01__01/generator.cpp",
            "generated",
            "available",
            f"equity_lsm_{model.time_kind}",
            model.model,
            product="american_option",
            variant=f"american_{side}s",
        )
        for model in AMERICAN_RECIPE_SPECS
        for side in ("call", "put")
    )


def _fixed_income_price_dataset_specs() -> tuple[DatasetSpec, ...]:
    result = []
    for model, curve, variants in FIXED_INCOME_CAPABILITIES:
        for variant in variants:
            components = [f"{model}_01"]
            if curve is not None:
                components.append(f"{curve}_01")
            components.extend((f"{variant}_01", "01"))
            dataset_id = "__".join(components)
            curve_path = f"{curve}/" if curve is not None else ""
            engine = (
                "fixed_income_lsm"
                if variant.startswith("bermudan_")
                else "fixed_income_closed_form"
            )
            result.append(DatasetSpec(
                dataset_id,
                "prices",
                "fixed_income",
                f"catalog/model/fixed_income/{model}/prices/{curve_path}"
                f"{variant}/{dataset_id}/generator.cpp",
                "declared_manual",
                "available",
                engine,
                model,
                curve,
                FIXED_INCOME_VARIANTS[variant],
                variant,
            ))
    return tuple(result)


def _sample_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            f"samples_{recipe_index:02d}",
            "samples",
            model.asset_class,
            f"catalog/model/{model.asset_class}/{model.name}/samples/"
            f"samples_{recipe_index:02d}/generator.cpp",
            "declared_manual",
            model.sample_recipe_status,
            model.sample_engine,
            model.name,
            condition=model.sample_requirement,
        )
        for model in MODEL_SPECS
        for recipe_index in (1, 2)
    )


DATASET_SPECS = (
    _model_parameter_dataset_specs()
    + _product_parameter_dataset_specs()
    + _curve_dataset_specs()
    + _ordinary_equity_price_dataset_specs()
    + _american_price_dataset_specs()
    + _fixed_income_price_dataset_specs()
    + _sample_dataset_specs()
)
AVAILABLE_DATASET_SPECS = tuple(
    dataset for dataset in DATASET_SPECS if dataset.status == "available"
)
DEFERRED_DATASET_SPECS = tuple(
    dataset for dataset in DATASET_SPECS if dataset.status == "deferred"
)


def resolve_price_capability(
    model: str,
    product: str,
    variant: str,
    curve: str | None = None,
) -> DatasetSpec:
    matches = [
        dataset for dataset in AVAILABLE_DATASET_SPECS
        if dataset.dataset_kind == "prices"
        and dataset.model == model
        and dataset.product == product
        and dataset.variant == variant
        and dataset.curve == curve
    ]
    if len(matches) != 1:
        raise KeyError(
            "unsupported or ambiguous pricing capability: "
            f"{model}/{curve or '-'}/{product}/{variant}"
        )
    return matches[0]


EQUITY_EARLY_EXERCISE_UNITS = tuple(
    f"{model.model}/product/american_option" for model in AMERICAN_RECIPE_SPECS
)
EQUITY_SAMPLE_UNITS = tuple(
    f"{model.name}/sample"
    for model in MODEL_SPECS
    if model.asset_class == "equity"
    and model.sample_binding_status == "available"
    and model.sample_requirement is None
)
EQUITY_MATHDX_SAMPLE_UNITS = tuple(
    f"{model.name}/sample"
    for model in MODEL_SPECS
    if model.asset_class == "equity"
    and model.sample_binding_status == "available"
    and model.sample_requirement == "AI_FACTORY_MATHDX_ROOT"
)


def _fixed_income_units() -> tuple[str, ...]:
    units = {
        f"{model.name}/sample"
        for model in MODEL_SPECS
        if model.asset_class == "fixed_income"
        and model.sample_binding_status == "available"
    }
    for model, curve, variants in FIXED_INCOME_CAPABILITIES:
        for variant in variants:
            product = FIXED_INCOME_VARIANTS[variant]
            curve_path = f"{curve}/" if curve is not None else ""
            units.add(f"{model}/product/{curve_path}{product}")
    return tuple(sorted(units))


FIXED_INCOME_UNITS = _fixed_income_units()


CAPABILITY_EXCEPTIONS = (
    CapabilityExceptionSpec(
        "CAP-FI-RECIPE-BODY",
        "the 58 declared fixed-income price recipe bodies",
        "tools/pricing fixed-income helpers",
        "curve schedules and analytical versus LSM inputs still use several "
        "reviewed shared helpers; their compatibility matrix is canonical here",
        "add a typed fixed-income renderer before adding a 59th recipe",
    ),
    CapabilityExceptionSpec(
        "CAP-PARAMETER-RECIPE-BODY",
        "the 53 declared model, product and curve parameter recipe bodies",
        "tools/datasets parameter-generation helpers",
        "parameter laws are domain-specific while paths and multiplicity are "
        "canonical here",
        "add a parameter-recipe renderer before introducing a second recipe "
        "for an existing owner",
    ),
)
