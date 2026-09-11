"""Typed repository capability matrix composed with the pricing manifest.

The C++ algorithms remain in ``src/``.  This module owns the published
compatibility matrix, dataset-recipe paths and CMake registration units,
including explicitly deferred capabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from pathlib import PurePosixPath

from manifest import (
    AMERICAN_RECIPE_SPECS,
    BINDINGS,
    BLACK_SCHOLES_CLOSED_FORM_PRODUCTS,
    Binding,
    MODEL_RECIPE_SPECS,
    PRICE_VARIANTS,
    ROUGH_N_FACTOR_MODELS,
    ROUGH_PRODUCT_BINDINGS,
    ROUGH_VOLTERRA_MODELS,
    RoughProductBinding,
)
from sample_manifest import SAMPLE_MODELS


SCHEMA_VERSION = 2

RNG_DOMAIN_VERSION = 3
RNG_NAMESPACE_BASE = 0xA1F0_0000_0000_0000
RNG_DOMAIN_STRIDE = 1 << 32
RNG_STREAM_CAPACITY = 1 << 30
RNG_COMMON_RANDOM_NUMBER_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()


@dataclass(frozen=True)
class SourceSymbol:
    path: str
    name: str


@dataclass(frozen=True)
class EngineSpec:
    name: str
    asset_class: str
    algorithm: str
    binding_template_family: str
    recipe_template_family: str
    schedule_contract: str
    transition_contract: str
    analytics_contract: str
    optional_dependency: str | None = None
    concepts: tuple[SourceSymbol, ...] = ()
    launchers: tuple[SourceSymbol, ...] = ()
    runners: tuple[SourceSymbol, ...] = ()
    instantiation_strategy: str = ""


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
    transition_contract: str
    analytics_contract: str
    state_contract: str
    observables: tuple[str, ...]
    supported_architectures: tuple[str, ...]
    sample_requirement: str | None = None
    renderer_alias: str | None = None
    rng_domain_epoch: int = 0

    @property
    def source_prefix(self) -> str:
        if self.asset_class == "fixed_income":
            return f"model/fixed_income/{self.name}"
        family = "rough" if self.family.startswith("rough_") else "markovian"
        return f"model/equity/{family}/{self.name}"


@dataclass(frozen=True)
class ProductSpec:
    name: str
    asset_class: str
    parameter_dataset_ids: tuple[str, ...]
    path_policy: str
    schedule_contract: str
    observation_contract: str
    exercise_contract: str
    sided: bool
    required_capabilities: tuple[str, ...]

    @property
    def source_prefix(self) -> str:
        return f"product/{self.name}"


@dataclass(frozen=True)
class CurveSpec:
    name: str
    display: str
    cpp_type: str
    parameter_dataset_id: str

    @property
    def source_prefix(self) -> str:
        return f"curve/{self.name}"


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    dataset_kind: str
    asset_class: str
    recipe_path: str
    owner: str
    status: str
    source_prefix: str
    template: str | None = None
    engine: str | None = None
    model: str | None = None
    curve: str | None = None
    product: str | None = None
    variant: str | None = None
    condition: str | None = None
    construction: str = ""
    numerical_profile: str = ""
    layout: str = ""

    @property
    def catalog_yaml_path(self) -> str:
        return str(PurePosixPath(self.recipe_path).with_name("dataset.yaml"))

    @property
    def dataset_path(self) -> str:
        recipe = PurePosixPath(self.recipe_path)
        relative = recipe.relative_to("catalog")
        return str(
            PurePosixPath("datasets")
            / relative.parent.parent
            / f"{self.dataset_id}.json"
        )

    @property
    def url(self) -> str:
        relative = PurePosixPath(self.dataset_path).relative_to("datasets")
        return f"https://datasets.ai-factory.example/v1/{relative}"

    @property
    def cmake_target(self) -> str:
        if self.dataset_kind in {"prices", "price_delta"}:
            components = self.dataset_id.split("__")[:-1]
            names = [component.rsplit("_", 1)[0] for component in components]
            version = self.dataset_id.split("__")[-1]
            return "generate_" + "_".join((*names, version))
        if self.dataset_kind == "samples":
            return f"generate_{self.model}_{self.dataset_id}"
        return f"generate_{self.dataset_id}"


@dataclass(frozen=True)
class RngDomainSpec:
    """Versioned, disjoint Philox-key reservation for one dataset recipe."""

    version: int
    recipe_path: str
    ordinal: int
    streams: tuple[str, ...]
    common_random_number_group: str | None = None

    @property
    def base_seed(self) -> int:
        return RNG_NAMESPACE_BASE + self.ordinal * RNG_DOMAIN_STRIDE

    def seed(self, stream: str) -> int:
        try:
            stream_index = self.streams.index(stream)
        except ValueError as error:
            raise KeyError(
                f"unknown RNG stream {stream!r} for {self.recipe_path}"
            ) from error
        return self.base_seed + stream_index * RNG_STREAM_CAPACITY

    def interval(self, stream: str) -> tuple[int, int]:
        """Return the half-open key interval reserved for ``stream``."""
        start = self.seed(stream)
        return start, start + RNG_STREAM_CAPACITY


@dataclass(frozen=True)
class FixedIncomeCapabilitySpec:
    model: str
    curve: str | None
    factorization: str
    implementation: str | None
    variants: tuple[str, ...]
    terminal_forward_dynamics: str | None = None

    @property
    def source_prefix(self) -> str:
        return f"model/fixed_income/{self.model}"

    @property
    def binding_template_family(self) -> str:
        suffix = f"/{self.implementation}" if self.implementation else ""
        return (
            f"pricing/closed_form/fixed_income/{self.factorization}{suffix}"
        )

    @property
    def recipe_template_family(self) -> str:
        suffix = f"/{self.implementation}" if self.implementation else ""
        return f"catalog/pricing/fixed_income/{self.factorization}{suffix}"

    def engine_for(self, product: str) -> str:
        if product == "bermudan_swaption":
            return "fixed_income_lsm"
        if product == "european_swaption" and self.factorization in {
            "affine_two_factor", "curve_fitted_two_factor"
        }:
            return "fixed_income_monte_carlo"
        return "fixed_income_closed_form"

    def binding_template_for(self, product: str) -> str:
        if product == "bermudan_swaption" and self.curve and self.terminal_forward_dynamics:
            return "pricing/longstaff_schwartz/fixed_income/terminal_forward/curve_fitted"
        if self.engine_for(product) == "fixed_income_monte_carlo":
            family = "curve_fitted" if self.curve else "standalone"
            return f"pricing/markovian/fixed_income/{family}"
        return self.binding_template_family

    def recipe_template_for(self, product: str) -> str:
        if product == "bermudan_swaption" and self.curve and self.terminal_forward_dynamics:
            return "catalog/pricing/fixed_income/longstaff_schwartz/terminal_forward/curve_fitted/bermudan_swaption.generator.cpp.tpl"
        if self.engine_for(product) == "fixed_income_monte_carlo":
            family = "curve_fitted" if self.curve else "standalone"
            return f"catalog/pricing/fixed_income/monte_carlo/{family}/{product}.generator.cpp.tpl"
        return f"{self.recipe_template_family}/{product}.generator.cpp.tpl"

    def is_generated(self, product: str) -> bool:
        return product != "bermudan_swaption" or bool(self.curve and self.terminal_forward_dynamics)


@dataclass(frozen=True)
class ProductBindingSpec:
    model: str
    asset_class: str
    product: str
    curve: str | None
    engine: str
    owner: str
    source_prefix: str
    template_family: str | None
    transition_contract: str
    manifest_binding: Binding | RoughProductBinding | None = None
    factorization: str | None = None
    implementation: str | None = None
    required_capabilities: tuple[str, ...] = ()
    terminal_forward_dynamics: str | None = None

    @property
    def unit_path(self) -> str:
        qualifier = f"/{self.curve}" if self.curve is not None else ""
        return (
            f"src/{self.source_prefix}/product{qualifier}/{self.product}"
        )

    @property
    def paths(self) -> tuple[str, str]:
        return (f"{self.unit_path}.cuh", f"{self.unit_path}.cu")


@dataclass(frozen=True)
class CapabilityExceptionSpec:
    identifier: str
    scope: str
    owner: str
    rationale: str
    closure_condition: str


@dataclass(frozen=True)
class ResolvedPriceCapability:
    engine: EngineSpec
    model: ModelSpec
    product: ProductSpec
    binding: ProductBindingSpec
    dataset: DatasetSpec

    @property
    def target(self) -> str:
        return self.dataset.cmake_target

    @property
    def recipe_path(self) -> str:
        return self.dataset.recipe_path


class PriceCapabilityState(Enum):
    AVAILABLE = "available"
    DEFERRED = "deferred"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class PriceCapabilityResolution:
    state: PriceCapabilityState
    matches: tuple[DatasetSpec, ...] = ()


def source_symbol(path: str, name: str) -> SourceSymbol:
    return SourceSymbol(path, name)


S = source_symbol


ENGINE_SPECS = (
    EngineSpec(
        "equity_closed_form", "equity", "closed-form CUDA",
        "pricing/closed_form/black_scholes",
        "catalog/pricing/black_scholes_closed_form",
        "analytical product contract", "no stochastic transition",
        "model analytical pricing policy",
        concepts=(S("src/common/closed_form/concepts.cuh", "closed_form::ClosedFormPricingPolicy"),),
        launchers=(S("src/common/closed_form/closed_form_kernels.cuh", "closed_form::launch_closed_form_cuda"),),
        runners=(S("tools/pricing/equity_price_generation.cuh", "offline::pricing::generate_analytical_equity_price_dataset"),),
        instantiation_strategy="product/side compile-time specialization",
    ),
    EngineSpec(
        "equity_markovian", "equity", "Markovian CUDA Monte Carlo",
        "pricing/markovian", "catalog/pricing/markovian",
        "terminal, dense, regular or two-date calendar",
        "exact or fixed-step model transition", "none",
        concepts=(S("src/common/monte_carlo/concepts.cuh", "monte_carlo::ScalarMonteCarloPricingPolicy"),),
        launchers=(S("src/common/monte_carlo/monte_carlo_kernel.cuh", "monte_carlo::launch_monte_carlo_cuda"),),
        runners=(S("tools/pricing/equity_price_generation.cuh", "offline::pricing::generate_monte_carlo_equity_price_dataset"),),
        instantiation_strategy="model/product/schedule/side specialization",
    ),
    EngineSpec(
        "equity_volterra_fft",
        "equity",
        "Gaussian-Volterra hybrid FFT CUDA Monte Carlo",
        "pricing/rough/volterra_fft",
        "catalog/pricing/rough/volterra_fft",
        "hybrid terminal, dense, regular or two-date calendar",
        "Gaussian-Volterra hybrid FFT",
        "none",
        "mathDx/cuFFTDx",
        concepts=(
            S("src/common/volterra/concepts.cuh", "volterra::HybridKernelPolicy"),
            S("src/common/volterra/concepts.cuh", "volterra::HybridPathPolicyFor"),
        ),
        launchers=(S("src/common/volterra/hybrid_fft_pricer.cuh", "volterra::hybrid_fft::launch_pricing_cuda"),),
        runners=(S("tools/pricing/equity_price_generation.cuh", "offline::pricing::generate_volterra_equity_price_dataset"),),
        instantiation_strategy="kernel/model/product/schedule specialization",
    ),
    EngineSpec(
        "equity_n_factor",
        "equity",
        "prepared N-factor Markovian CUDA Monte Carlo",
        "pricing/rough/markovian_n_factor",
        "catalog/pricing/rough/markovian_n_factor",
        "fixed-step terminal, dense, regular or two-date calendar",
        "prepared 2/3/7-factor Markovian lift",
        "none",
        concepts=(S("src/common/monte_carlo/concepts.cuh", "monte_carlo::ScalarMonteCarloPricingPolicy"),),
        launchers=(S("src/common/equity/prepared_path_product_pricing.cuh", "equity::launch_prepared_path_product_cuda"),),
        runners=(S("tools/pricing/equity_price_generation.cuh", "offline::pricing::generate_monte_carlo_equity_price_dataset"),),
        instantiation_strategy="factor-count/product/schedule specialization",
    ),
    EngineSpec(
        "equity_lsm_fixed",
        "equity",
        "fixed-step Longstaff-Schwartz CUDA",
        "hand_written/equity/american_option",
        "catalog/pricing/american_longstaff_schwartz",
        "explicit American exercise calendar",
        "fixed-step model transition",
        "Longstaff-Schwartz regression",
        concepts=(S("src/common/longstaff_schwartz/concepts.cuh", "longstaff_schwartz::LongstaffSchwartzPolicy"),),
        launchers=(S("src/common/longstaff_schwartz/longstaff_schwartz_kernels.cuh", "longstaff_schwartz::launch_longstaff_schwartz_cuda"),),
        runners=(S("tools/pricing/american_option_price_generation.cuh", "offline::pricing::generate_american_option_equity_price_dataset"),),
        instantiation_strategy="model/side fixed-step specialization",
    ),
    EngineSpec(
        "equity_lsm_exact",
        "equity",
        "exact-transition Longstaff-Schwartz CUDA",
        "hand_written/equity/american_option",
        "catalog/pricing/american_longstaff_schwartz",
        "explicit American exercise calendar",
        "exact model transition",
        "Longstaff-Schwartz regression",
        concepts=(S("src/common/longstaff_schwartz/concepts.cuh", "longstaff_schwartz::LongstaffSchwartzPolicy"),),
        launchers=(S("src/common/longstaff_schwartz/longstaff_schwartz_kernels.cuh", "longstaff_schwartz::launch_longstaff_schwartz_cuda"),),
        runners=(S("tools/pricing/american_option_price_generation.cuh", "offline::pricing::generate_american_option_equity_price_dataset"),),
        instantiation_strategy="model/side exact-transition specialization",
    ),
    EngineSpec(
        "fixed_income_closed_form",
        "fixed_income",
        "fixed-income closed-form CUDA",
        "pricing/closed_form/fixed_income",
        "catalog/pricing/fixed_income",
        "contractual rate-option, bond-option or swaption schedule",
        "no stochastic transition",
        "standalone or curve-fitted affine provider",
        concepts=(S("src/common/closed_form/concepts.cuh", "closed_form::ClosedFormPricingPolicy"),),
        launchers=(S("src/common/closed_form/closed_form_kernels.cuh", "closed_form::launch_closed_form_cuda"),),
        runners=(S("tools/pricing/european_swaption_price_generation.cuh", "datasets::generate_regular_european_swaption_prices"),),
        instantiation_strategy="provider/product/side specialization",
    ),
    EngineSpec(
        "fixed_income_lsm",
        "fixed_income",
        "fixed-income Longstaff-Schwartz CUDA",
        "pricing/longstaff_schwartz/fixed_income",
        "catalog/pricing/fixed_income/longstaff_schwartz",
        "regular co-terminal Bermudan schedule",
        "exact joint state/integral or terminal-forward state transition",
        "Longstaff-Schwartz regression",
        concepts=(S("src/common/longstaff_schwartz/concepts.cuh", "longstaff_schwartz::LongstaffSchwartzPolicy"),),
        launchers=(S("src/common/longstaff_schwartz/longstaff_schwartz_kernels.cuh", "longstaff_schwartz::launch_longstaff_schwartz_cuda"),),
        runners=(S("tools/pricing/bermudan_swaption_price_generation.cuh", "datasets::generate_bermudan_swaption_prices"),),
        instantiation_strategy="model/curve/side transition specialization",
    ),
    EngineSpec(
        "fixed_income_monte_carlo", "fixed_income", "terminal exact-transition Monte Carlo CUDA",
        "pricing/markovian/fixed_income", "catalog/pricing/fixed_income/monte_carlo",
        "single contractual exercise date", "exact joint factor/integral transition under Q",
        "conditional zero-coupon bonds and path discount factor",
        concepts=(S("src/common/monte_carlo/concepts.cuh", "monte_carlo::ScalarMonteCarloPricingPolicy"),),
        launchers=(S("src/common/monte_carlo/monte_carlo_kernel.cuh", "monte_carlo::launch_monte_carlo_cuda"),),
        runners=(S("tools/pricing/european_swaption_monte_carlo_generation.cuh", "datasets::generate_european_swaption_monte_carlo_prices"),),
        instantiation_strategy="joint dynamics/schedule/model composition/product/side specialization",
    ),
    EngineSpec(
        "sample_markovian", "equity", "Markovian CUDA sampling",
        "sampling/markovian", "sampling/catalog",
        "random terminal maturity", "exact or fixed-step model transition",
        "model-only observation",
        concepts=(S("src/common/sample/concepts.cuh", "sample::SamplingPolicy"),),
        launchers=(S("src/common/sample/sample_kernels.cuh", "sample::launch_samples_cuda"),),
        runners=(S("tools/sampling/model_sample_generation.cuh", "offline::sampling::generate_model_sample_dataset"),),
        instantiation_strategy="model/schedule/observation specialization",
    ),
    EngineSpec(
        "sample_n_factor", "equity", "N-factor CUDA sampling",
        "sampling/rough/markovian_n_factor", "sampling/catalog",
        "random terminal maturity", "prepared seven-factor lift",
        "model-only observation",
        concepts=(S("src/common/sample/concepts.cuh", "sample::SamplingPolicy"),),
        launchers=(S("src/common/sample/sample_kernels.cuh", "sample::launch_samples_cuda"),),
        runners=(S("tools/sampling/model_sample_generation.cuh", "offline::sampling::generate_prepared_model_sample_dataset"),),
        instantiation_strategy="model/factor-count specialization",
    ),
    EngineSpec(
        "sample_volterra_fft",
        "equity",
        "Gaussian-Volterra hybrid FFT CUDA sampling",
        "sampling/rough/volterra_fft",
        "sampling/catalog",
        "random terminal maturity",
        "Gaussian-Volterra hybrid FFT",
        "model-only observation",
        "mathDx/cuFFTDx",
        concepts=(
            S("src/common/volterra/concepts.cuh", "volterra::HybridKernelPolicy"),
            S("src/common/sample/concepts.cuh", "sample::VolterraFftSamplingPolicy"),
        ),
        launchers=(S("src/common/sample/volterra_fft_sample_kernels.cuh", "sample::volterra_fft::launch_samples_cuda"),),
        runners=(S("tools/sampling/model_sample_generation.cuh", "offline::sampling::generate_model_sample_dataset"),),
        instantiation_strategy="kernel/model/observation specialization",
    ),
    EngineSpec(
        "sample_fixed_income",
        "fixed_income",
        "fixed-income CUDA sampling",
        "sampling/markovian", "sampling/catalog",
        "random terminal maturity", "exact model transition",
        "model-only observation",
        concepts=(S("src/common/sample/concepts.cuh", "sample::SamplingPolicy"),),
        launchers=(S("src/common/sample/sample_kernels.cuh", "sample::launch_samples_cuda"),),
        runners=(S("tools/sampling/model_sample_generation.cuh", "offline::sampling::generate_model_sample_dataset"),),
        instantiation_strategy="model/observation specialization",
    ),
)
ENGINE_BY_NAME = {engine.name: engine for engine in ENGINE_SPECS}


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


def _model_transition_contract(model) -> str:
    if model.backend == "volterra":
        return "Gaussian-Volterra hybrid FFT"
    if model.backend == "n_factor":
        return "prepared fixed-step 2/3/7-factor Markovian lift"
    if model.time_kind == "exact":
        return "exact transition"
    return "fixed-step transition"


MODEL_SPECS = tuple(
    ModelSpec(
        name=model.name,
        display=model.display,
        asset_class=model.asset_class,
        family=(
            _equity_model_family(model.name)
            if model.asset_class == "equity" else "markovian"
        ),
        parameter_dataset_id=f"{model.name}_01",
        pricing_engine=(
            _equity_pricing_engine(model.backend)
            if model.asset_class == "equity"
            else "fixed_income_closed_form"
        ),
        sample_engine=(
            _equity_sample_engine(model.name)
            if model.asset_class == "equity"
            else "sample_fixed_income"
        ),
        sample_binding_status="available",
        sample_recipe_status="available",
        transition_contract=_model_transition_contract(model),
        analytics_contract=model.analytics_contract,
        state_contract=model.observation,
        observables=model.outputs,
        supported_architectures=model.supported_architectures,
        sample_requirement=(
            "AI_FACTORY_MATHDX_ROOT"
            if model.name in ROUGH_VOLTERRA_NAMES else None
        ),
        renderer_alias=model.template_alias,
        rng_domain_epoch=model.rng_domain_epoch,
    )
    for model in SAMPLE_MODELS
)
MODEL_BY_NAME = {model.name: model for model in MODEL_SPECS}


def derive_equity_product_specs(
    variants=PRICE_VARIANTS,
    product_bindings=ROUGH_PRODUCT_BINDINGS,
) -> tuple[ProductSpec, ...]:
    datasets: dict[str, set[str]] = {}
    for variant in variants:
        datasets.setdefault(variant.product, set()).add(
            variant.product_dataset_id
        )
    bindings = {
        binding.product: binding for binding in product_bindings
    }
    if len(bindings) != len(tuple(product_bindings)):
        raise ValueError("duplicate canonical equity product contract")
    missing_bindings = sorted(set(datasets) - set(bindings))
    missing_variants = sorted(set(bindings) - set(datasets))
    if missing_bindings or missing_variants:
        raise ValueError(
            "equity product contract/variant mismatch: "
            f"missing bindings={missing_bindings}, "
            f"missing variants={missing_variants}"
        )
    result = [
        ProductSpec(
            product,
            "equity",
            tuple(sorted(dataset_ids)),
            bindings[product].path_policy,
            bindings[product].schedule_kind,
            bindings[product].observation_coordinate,
            bindings[product].exercise_contract,
            bindings[product].sided,
            bindings[product].required_capabilities,
        )
        for product, dataset_ids in sorted(datasets.items())
    ]
    result.append(ProductSpec(
        "american_option",
        "equity",
        ("american_options_01",),
        "AmericanOptionPricingPolicy",
        "explicit exercise calendar",
        "spot and continuation state",
        "american",
        True,
        ("spot", "continuation_state", "discount_factor"),
    ))
    return tuple(sorted(result, key=lambda product: product.name))


PRODUCT_SPECS = derive_equity_product_specs() + (
    ProductSpec(
        "bermudan_swaption",
        "fixed_income",
        ("bermudan_swaptions_01",),
        "BermudanSwaptionPricingPolicy",
        "regular co-terminal exercise schedule",
        "continuation state under the binding-specific pricing measure",
        "bermudan",
        True,
        ("continuation_state", "zero_coupon_bond"),
    ),
    ProductSpec(
        "european_swaption",
        "fixed_income",
        ("european_swaptions_01",),
        "EuropeanSwaptionClosedFormPricingPolicy or EuropeanSwaptionMonteCarloPricingPolicyCore",
        "contractual fixed-leg schedule",
        "analytical provider or exact joint state/integral terminal Monte Carlo",
        "none",
        True,
        ("zero_coupon_bond",),
    ),
    ProductSpec(
        "rate_option",
        "fixed_income",
        ("rate_options_01",),
        "RateOptionClosedFormPricingPolicy",
        "fixing and payment dates",
        "analytical provider",
        "none",
        True,
        ("zero_coupon_bond", "bond_option"),
    ),
    ProductSpec(
        "zero_coupon_bond_option",
        "fixed_income",
        ("zero_coupon_bond_options_01",),
        "ZeroCouponBondOptionClosedFormPricingPolicy",
        "option expiry and bond maturity",
        "analytical provider",
        "none",
        True,
        ("zero_coupon_bond", "bond_option"),
    ),
)
PRODUCT_BY_NAME = {
    (product.asset_class, product.name): product
    for product in PRODUCT_SPECS
}


CURVE_SPECS = (
    CurveSpec(
        "nelson_siegel", "Nelson-Siegel", "NelsonSiegel", "nelson_siegel_01"
    ),
    CurveSpec("svensson", "Svensson", "Svensson", "svensson_01"),
)
CURVE_BY_NAME = {curve.name: curve for curve in CURVE_SPECS}


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
FIXED_INCOME_CAPABILITIES = (
    FixedIncomeCapabilitySpec(
        "cir", None, "affine_one_factor", "cir",
        FIXED_INCOME_ALL_VARIANTS,
        terminal_forward_dynamics="cir",
    ),
    FixedIncomeCapabilitySpec(
        "cir_plus_plus", "nelson_siegel", "curve_fitted_one_factor", None,
        FIXED_INCOME_ALL_VARIANTS,
        terminal_forward_dynamics="cir",
    ),
    FixedIncomeCapabilitySpec(
        "cir_plus_plus", "svensson", "curve_fitted_one_factor", None,
        FIXED_INCOME_ALL_VARIANTS,
        terminal_forward_dynamics="cir",
    ),
    FixedIncomeCapabilitySpec(
        "g2", None, "affine_two_factor", None,
        FIXED_INCOME_ALL_VARIANTS,
    ),
    FixedIncomeCapabilitySpec(
        "g2_plus_plus", "nelson_siegel", "curve_fitted_two_factor", None,
        FIXED_INCOME_ALL_VARIANTS,
    ),
    FixedIncomeCapabilitySpec(
        "g2_plus_plus", "svensson", "curve_fitted_two_factor", None,
        FIXED_INCOME_ALL_VARIANTS,
    ),
    FixedIncomeCapabilitySpec(
        "hull_white", "nelson_siegel", "curve_fitted_one_factor", None,
        FIXED_INCOME_ALL_VARIANTS,
    ),
    FixedIncomeCapabilitySpec(
        "hull_white", "svensson", "curve_fitted_one_factor", None,
        FIXED_INCOME_ALL_VARIANTS,
    ),
    FixedIncomeCapabilitySpec(
        "ornstein_uhlenbeck", None, "affine_one_factor", "gaussian",
        FIXED_INCOME_ALL_VARIANTS,
    ),
    FixedIncomeCapabilitySpec(
        "vasicek", None, "affine_one_factor", "gaussian",
        FIXED_INCOME_ALL_VARIANTS,
    ),
)


def _model_parameter_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset_id=model.parameter_dataset_id,
            dataset_kind="model_parameters",
            asset_class=model.asset_class,
            recipe_path=(
                f"catalog/{model.source_prefix}/parameters/"
                f"{model.parameter_dataset_id}/generator.cpp"
            ),
            owner="hand_written",
            status="available",
            source_prefix=model.source_prefix,
            model=model.name,
            construction="ordered_core_stress",
            numerical_profile="fp32_parameter_rows_90_10",
            layout="row_major_parameter_records",
        )
        for model in MODEL_SPECS
    )


def _product_parameter_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset_id=dataset_id,
            dataset_kind="product_parameters",
            asset_class=product.asset_class,
            recipe_path=(
                f"catalog/{product.source_prefix}/{dataset_id}/generator.cpp"
            ),
            owner="hand_written",
            status="available",
            source_prefix=product.source_prefix,
            product=product.name,
            construction="ordered_core_stress",
            numerical_profile="fp32_parameter_rows_90_10",
            layout="row_major_parameter_records",
        )
        for product in PRODUCT_SPECS
        for dataset_id in product.parameter_dataset_ids
    )


def _curve_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset_id=curve.parameter_dataset_id,
            dataset_kind="curve_parameters",
            asset_class="fixed_income",
            recipe_path=(
                f"catalog/{curve.source_prefix}/"
                f"{curve.parameter_dataset_id}/generator.cpp"
            ),
            owner="hand_written",
            status="available",
            source_prefix=curve.source_prefix,
            curve=curve.name,
            construction="ordered_core_stress",
            numerical_profile="fp32_parameter_rows_90_10",
            layout="row_major_parameter_records",
        )
        for curve in CURVE_SPECS
    )


def _ordinary_equity_price_dataset_specs() -> tuple[DatasetSpec, ...]:
    result = []
    for model in MODEL_RECIPE_SPECS:
        model_spec = MODEL_BY_NAME[model.name]
        for variant in PRICE_VARIANTS:
            dataset_id = f"{model.name}_01__{variant.name}_01__01"
            engine = _equity_pricing_engine(model.backend)
            if (
                model.name == "black_scholes"
                and variant.product in BLACK_SCHOLES_CLOSED_FORM_PRODUCTS
            ):
                engine = "equity_closed_form"
            result.append(DatasetSpec(
                dataset_id=dataset_id,
                dataset_kind="prices",
                asset_class="equity",
                recipe_path=(
                    f"catalog/{model_spec.source_prefix}/prices/"
                    f"{variant.name}/{dataset_id}/generator.cpp"
                ),
                owner="generated",
                status="available",
                source_prefix=model_spec.source_prefix,
                template=(
                    ENGINE_BY_NAME[engine].recipe_template_family
                    + "/generator.cpp.tpl"
                ),
                engine=engine,
                model=model.name,
                product=variant.product,
                variant=variant.name,
                condition=(
                    "AI_FACTORY_MATHDX_ROOT"
                    if engine == "equity_volterra_fft" else None
                ),
                construction="aligned",
                numerical_profile="catalog_price_profile",
                layout="one_price_per_aligned_input_row",
            ))
    return tuple(result)


def _american_price_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset_id=f"{model.model}_01__american_{side}s_01__01",
            dataset_kind="prices",
            asset_class="equity",
            recipe_path=(
                f"catalog/{MODEL_BY_NAME[model.model].source_prefix}/prices/"
                f"american_{side}s/"
                f"{model.model}_01__american_{side}s_01__01/generator.cpp"
            ),
            owner="generated",
            status="available",
            source_prefix=MODEL_BY_NAME[model.model].source_prefix,
            template=(
                "catalog/pricing/american_longstaff_schwartz/"
                "generator.cpp.tpl"
            ),
            engine=f"equity_lsm_{model.time_kind}",
            model=model.model,
            product="american_option",
            variant=f"american_{side}s",
            construction="aligned",
            numerical_profile="longstaff_schwartz_price_profile",
            layout="one_price_per_aligned_input_row",
        )
        for model in AMERICAN_RECIPE_SPECS
        for side in ("call", "put")
    )


def _fixed_income_price_dataset_specs() -> tuple[DatasetSpec, ...]:
    result = []
    for capability in FIXED_INCOME_CAPABILITIES:
        for variant in capability.variants:
            components = [f"{capability.model}_01"]
            if capability.curve is not None:
                components.append(f"{capability.curve}_01")
            components.extend((f"{variant}_01", "01"))
            dataset_id = "__".join(components)
            curve_path = (
                f"{capability.curve}/"
                if capability.curve is not None else ""
            )
            engine = capability.engine_for(FIXED_INCOME_VARIANTS[variant])
            result.append(DatasetSpec(
                dataset_id=dataset_id,
                dataset_kind="prices",
                asset_class="fixed_income",
                recipe_path=(
                    f"catalog/{capability.source_prefix}/prices/{curve_path}"
                    f"{variant}/{dataset_id}/generator.cpp"
                ),
                owner=(
                    "generated" if capability.is_generated(FIXED_INCOME_VARIANTS[variant])
                    else "hand_written"
                ),
                status="available",
                source_prefix=capability.source_prefix,
                template=(
                    None
                    if not capability.is_generated(FIXED_INCOME_VARIANTS[variant])
                    else capability.recipe_template_for(FIXED_INCOME_VARIANTS[variant])
                ),
                engine=engine,
                model=capability.model,
                curve=capability.curve,
                product=FIXED_INCOME_VARIANTS[variant],
                variant=variant,
                construction="aligned",
                numerical_profile=(
                    "longstaff_schwartz_price_profile"
                    if engine == "fixed_income_lsm"
                    else "exact_joint_monte_carlo_price_profile"
                    if engine == "fixed_income_monte_carlo"
                    else "closed_form_price_profile"
                ),
                layout="one_price_per_aligned_input_row",
            ))
    return tuple(result)


def _sample_dataset_specs() -> tuple[DatasetSpec, ...]:
    return tuple(
        DatasetSpec(
            dataset_id=f"samples_{recipe_index:02d}",
            dataset_kind="samples",
            asset_class=model.asset_class,
            recipe_path=(
                f"catalog/{model.source_prefix}/samples/"
                f"samples_{recipe_index:02d}/generator.cpp"
            ),
            owner="generated",
            status=model.sample_recipe_status,
            source_prefix=model.source_prefix,
            template="sampling/catalog/generator.cpp.tpl",
            engine=model.sample_engine,
            model=model.name,
            condition=model.sample_requirement,
            construction="cartesian_parameter_paths",
            numerical_profile="model_sample_profile",
            layout="parameter_major_trajectory_rows",
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


_STOCHASTIC_PRICING_ENGINES = frozenset({
    "equity_markovian",
    "equity_volterra_fft",
    "equity_n_factor",
    "equity_lsm_fixed",
    "equity_lsm_exact",
    "fixed_income_lsm",
    "fixed_income_monte_carlo",
})


def _uses_philox(dataset: DatasetSpec) -> bool:
    return (
        dataset.dataset_kind == "samples"
        or dataset.engine in _STOCHASTIC_PRICING_ENGINES
    )


def _rng_streams(dataset: DatasetSpec) -> tuple[str, ...]:
    if dataset.dataset_kind == "samples":
        return ("parameters", "schedule", "dynamics")
    return ("dynamics",)


RNG_DOMAIN_SPECS = tuple(
    RngDomainSpec(
        version=RNG_DOMAIN_VERSION,
        recipe_path=dataset.recipe_path,
        ordinal=ordinal,
        streams=_rng_streams(dataset),
    )
    for ordinal, dataset in enumerate(
        sorted(
            (
                dataset for dataset in AVAILABLE_DATASET_SPECS
                if _uses_philox(dataset)
            ),
            # Append extension epochs; preserve the frozen V1 and V2 prefixes.
            key=lambda dataset: (
                max(MODEL_BY_NAME[dataset.model].rng_domain_epoch,
                    int(dataset.engine == "fixed_income_monte_carlo")),
                dataset.recipe_path,
            ),
        )
    )
)
RNG_DOMAIN_BY_RECIPE = {
    domain.recipe_path: domain for domain in RNG_DOMAIN_SPECS
}


def validate_rng_domain_specs(
    domains: tuple[RngDomainSpec, ...],
    common_random_number_allowlist: frozenset[tuple[str, str]] | None = None,
) -> None:
    """Reject ambiguous, overlapping or out-of-range Philox reservations."""
    if common_random_number_allowlist is None:
        common_random_number_allowlist = RNG_COMMON_RANDOM_NUMBER_ALLOWLIST
    recipe_paths = [domain.recipe_path for domain in domains]
    if len(recipe_paths) != len(set(recipe_paths)):
        raise ValueError("duplicate RNG-domain recipe path")

    intervals: list[tuple[int, int, str, str]] = []
    for domain in domains:
        if domain.version != RNG_DOMAIN_VERSION:
            raise ValueError(
                f"unsupported RNG-domain version {domain.version}: "
                f"{domain.recipe_path}"
            )
        if not domain.streams or len(domain.streams) != len(set(domain.streams)):
            raise ValueError(
                f"empty or duplicate RNG streams: {domain.recipe_path}"
            )
        if len(domain.streams) * RNG_STREAM_CAPACITY > RNG_DOMAIN_STRIDE:
            raise ValueError(
                f"RNG streams exceed domain stride: {domain.recipe_path}"
            )
        for stream in domain.streams:
            start, end = domain.interval(stream)
            if end > 1 << 64:
                raise ValueError(
                    f"RNG interval exceeds uint64: {domain.recipe_path}"
                )
            intervals.append((start, end, domain.recipe_path, stream))

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        previous_start, previous_end, previous_path, previous_stream = previous
        current_start, current_end, current_path, current_stream = current
        if previous_end <= current_start:
            continue
        pair = tuple(sorted((previous_path, current_path)))
        if pair not in common_random_number_allowlist:
            raise ValueError(
                "overlapping RNG intervals: "
                f"{previous_path}:{previous_stream} "
                f"[{previous_start}, {previous_end}) and "
                f"{current_path}:{current_stream} "
                f"[{current_start}, {current_end})"
            )


def resolve_rng_domain(dataset: DatasetSpec | str) -> RngDomainSpec:
    recipe_path = (
        dataset.recipe_path if isinstance(dataset, DatasetSpec) else dataset
    )
    try:
        return RNG_DOMAIN_BY_RECIPE[recipe_path]
    except KeyError as error:
        raise KeyError(
            f"dataset has no declared Philox domain: {recipe_path}"
        ) from error


validate_rng_domain_specs(RNG_DOMAIN_SPECS)


def validate_dataset_spec(dataset: DatasetSpec) -> None:
    """Reject path, owner and renderer metadata that diverge from one owner."""
    expected_prefix = f"catalog/{dataset.source_prefix}/"
    if not dataset.recipe_path.startswith(expected_prefix):
        raise ValueError(
            f"recipe path does not inherit source prefix "
            f"{dataset.source_prefix}: {dataset.recipe_path}"
        )
    if dataset.owner not in {
        "generated", "hand_written", "explicit_exception", "unsupported",
        "deferred",
    }:
        raise ValueError(
            f"unknown artifact owner {dataset.owner}: {dataset.recipe_path}"
        )
    if dataset.owner == "generated" and dataset.template is None:
        raise ValueError(
            f"generated dataset lacks a template: {dataset.recipe_path}"
        )
    if dataset.owner != "generated" and dataset.template is not None:
        raise ValueError(
            f"non-generated dataset declares a renderer: {dataset.recipe_path}"
        )
    if dataset.construction not in {
        "ordered_core_stress", "aligned", "cartesian_parameter_paths",
    }:
        raise ValueError(
            f"unknown dataset construction {dataset.construction}: "
            f"{dataset.recipe_path}"
        )
    if not dataset.numerical_profile or not dataset.layout:
        raise ValueError(
            f"dataset lacks numerical profile or layout: {dataset.recipe_path}"
        )


for _dataset in DATASET_SPECS:
    validate_dataset_spec(_dataset)


def classify_price_capability(
    model: str,
    product: str,
    variant: str,
    curve: str | None = None,
    datasets: tuple[DatasetSpec, ...] = DATASET_SPECS,
) -> PriceCapabilityResolution:
    matches = [
        dataset for dataset in datasets
        if dataset.dataset_kind == "prices"
        and dataset.model == model
        and dataset.product == product
        and dataset.variant == variant
        and dataset.curve == curve
    ]
    if len(matches) > 1:
        return PriceCapabilityResolution(
            PriceCapabilityState.AMBIGUOUS,
            tuple(matches),
        )
    if matches:
        state = {
            "available": PriceCapabilityState.AVAILABLE,
            "deferred": PriceCapabilityState.DEFERRED,
            "unsupported": PriceCapabilityState.UNSUPPORTED,
        }.get(matches[0].status, PriceCapabilityState.UNCLASSIFIED)
        return PriceCapabilityResolution(state, tuple(matches))

    model_spec = MODEL_BY_NAME.get(model)
    known_variants = {candidate.name for candidate in PRICE_VARIANTS} | set(
        FIXED_INCOME_VARIANTS
    )
    if (
        model_spec is None
        or (model_spec.asset_class, product) not in PRODUCT_BY_NAME
        or variant not in known_variants
        or curve not in {None, *CURVE_BY_NAME}
    ):
        return PriceCapabilityResolution(PriceCapabilityState.UNCLASSIFIED)
    return PriceCapabilityResolution(PriceCapabilityState.UNSUPPORTED)


def resolve_price_capability(
    model: str,
    product: str,
    variant: str,
    curve: str | None = None,
) -> DatasetSpec:
    resolution = classify_price_capability(model, product, variant, curve)
    if resolution.state is PriceCapabilityState.AVAILABLE:
        return resolution.matches[0]
    identity = f"{model}/{curve or '-'}/{product}/{variant}"
    raise KeyError(f"{resolution.state.value} pricing capability: {identity}")


def resolve_complete_price_capability(
    model: str,
    product: str,
    variant: str,
    curve: str | None = None,
) -> ResolvedPriceCapability:
    """Resolve one published tuple through engine, binding, target and recipe."""
    dataset = resolve_price_capability(model, product, variant, curve)
    binding_matches = [
        binding for binding in PRODUCT_BINDING_SPECS
        if binding.model == model
        and binding.product == product
        and binding.curve == curve
        and binding.engine == dataset.engine
    ]
    if len(binding_matches) != 1:
        raise KeyError(
            "pricing capability lacks one unique binding: "
            f"{model}/{curve or '-'}/{product}/{variant}"
        )
    model_spec = MODEL_BY_NAME[model]
    return ResolvedPriceCapability(
        engine=ENGINE_BY_NAME[dataset.engine],
        model=model_spec,
        product=PRODUCT_BY_NAME[(model_spec.asset_class, product)],
        binding=binding_matches[0],
        dataset=dataset,
    )


def validate_price_capability_graph(
    datasets: tuple[DatasetSpec, ...],
    bindings: tuple[ProductBindingSpec, ...],
) -> None:
    """Require every published price recipe to own one exact composition."""
    for dataset in datasets:
        if dataset.dataset_kind != "prices" or dataset.status != "available":
            continue
        matches = [
            binding for binding in bindings
            if binding.model == dataset.model
            and binding.product == dataset.product
            and binding.curve == dataset.curve
            and binding.engine == dataset.engine
        ]
        if len(matches) != 1:
            raise ValueError(
                "published price recipe must resolve to exactly one binding: "
                f"{dataset.recipe_path} resolved {len(matches)}"
            )


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
    for capability in FIXED_INCOME_CAPABILITIES:
        for variant in capability.variants:
            product = FIXED_INCOME_VARIANTS[variant]
            curve_path = (
                f"{capability.curve}/"
                if capability.curve is not None else ""
            )
            units.add(
                f"{capability.model}/product/{curve_path}{product}"
            )
    return tuple(sorted(units))


FIXED_INCOME_UNITS = _fixed_income_units()


def _product_binding_specs() -> tuple[ProductBindingSpec, ...]:
    result: list[ProductBindingSpec] = []
    result.extend(
        ProductBindingSpec(
            binding.model,
            "equity",
            binding.product,
            None,
            "equity_markovian",
            "generated",
            MODEL_BY_NAME[binding.model].source_prefix,
            "pricing/markovian",
            f"{binding.time_kind} model transition",
            binding,
        )
        for binding in BINDINGS
    )
    result.extend(
        ProductBindingSpec(
            "black_scholes",
            "equity",
            product,
            None,
            "equity_closed_form",
            "generated",
            MODEL_BY_NAME["black_scholes"].source_prefix,
            "pricing/closed_form/black_scholes",
            "no stochastic transition",
        )
        for product in BLACK_SCHOLES_CLOSED_FORM_PRODUCTS
    )
    for engine, models, template_family in (
        (
            "equity_volterra_fft",
            ROUGH_VOLTERRA_MODELS,
            "pricing/rough/volterra_fft",
        ),
        (
            "equity_n_factor",
            ROUGH_N_FACTOR_MODELS,
            "pricing/rough/markovian_n_factor",
        ),
    ):
        result.extend(
            ProductBindingSpec(
                model,
                "equity",
                binding.product,
                None,
                engine,
                "generated",
                MODEL_BY_NAME[model].source_prefix,
                template_family,
                (
                    "Gaussian-Volterra hybrid FFT"
                    if engine == "equity_volterra_fft"
                    else "prepared fixed-step N-factor transition"
                ),
                binding,
            )
            for model, _ in models
            for binding in ROUGH_PRODUCT_BINDINGS
        )
    american_models = {
        model.model: (
            "equity_lsm_exact"
            if model.time_kind == "exact" else "equity_lsm_fixed"
        )
        for model in AMERICAN_RECIPE_SPECS
    }
    american_models["black_scholes"] = "equity_lsm_exact"
    result.extend(
        ProductBindingSpec(
            model,
            "equity",
            "american_option",
            None,
            engine,
            "hand_written",
            MODEL_BY_NAME[model].source_prefix,
            None,
            (
                "exact model transition"
                if engine == "equity_lsm_exact"
                else "fixed-step model transition"
            ),
        )
        for model, engine in sorted(american_models.items())
    )
    for capability in FIXED_INCOME_CAPABILITIES:
        for product in sorted({
            FIXED_INCOME_VARIANTS[variant]
            for variant in capability.variants
        }):
            early_exercise = product == "bermudan_swaption"
            result.append(ProductBindingSpec(
                capability.model,
                "fixed_income",
                product,
                capability.curve,
                capability.engine_for(product),
                "generated" if capability.is_generated(product) else "hand_written",
                capability.source_prefix,
                (
                    None
                    if not capability.is_generated(product)
                    else capability.binding_template_for(product)
                ),
                (
                    "exact terminal-forward state transition (last exercise bond numeraire)"
                    if early_exercise and capability.terminal_forward_dynamics
                    else "exact joint state/integral transition"
                    if early_exercise or capability.engine_for(product) == "fixed_income_monte_carlo"
                    else "no stochastic transition"
                ),
                factorization=capability.factorization,
                implementation=capability.implementation,
                required_capabilities=(
                    ("terminal_forward_state", "bond_numeraire", "zero_coupon_bond")
                    if early_exercise and capability.terminal_forward_dynamics
                    else ("joint_state_integral", "discount_factor", "zero_coupon_bond")
                    if early_exercise or capability.engine_for(product) == "fixed_income_monte_carlo"
                    else ("zero_coupon_bond", "bond_option", "jamshidian")
                    if product == "european_swaption"
                    else ("zero_coupon_bond", "bond_option")
                ),
                terminal_forward_dynamics=capability.terminal_forward_dynamics,
            ))
    keys = [spec.unit_path for spec in result]
    if len(keys) != len(set(keys)):
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        raise ValueError(
            f"duplicate ProductBindingSpec unit paths: {duplicates}"
        )
    return tuple(result)


PRODUCT_BINDING_SPECS = _product_binding_specs()


@dataclass(frozen=True)
class PriceDeltaBindingSpec:
    """A spot sensitivity of an existing pricing contract, not a new product."""

    pricing: ProductBindingSpec
    path_strategy: str

    def __post_init__(self) -> None:
        if self.path_strategy not in {"multiplicative", "coupled", "closed_form_bump"}:
            raise ValueError(f"Unknown price-delta strategy: {self.path_strategy}")
        expected_engines = ({"equity_closed_form"} if self.path_strategy == "closed_form_bump"
                            else {"equity_markovian", "equity_lsm_exact", "equity_lsm_fixed", "equity_n_factor", "equity_volterra_fft"})
        if self.pricing.asset_class != "equity" or self.pricing.engine not in expected_engines:
            raise ValueError("Price-delta strategy is incompatible with the pricing engine")

    @property
    def unit_path(self) -> str:
        return f"{self.pricing.unit_path}_price_delta"

    @property
    def paths(self) -> tuple[str, str]:
        return (f"{self.unit_path}.cuh", f"{self.unit_path}.cu")


# Each supported equity pricing contract has a separate price-delta launcher.
# State adapters preserve the model parameter conventions.
PRICE_DELTA_PATH_STRATEGIES = {
    "bates": "multiplicative", "black_scholes": "multiplicative",
    "cev": "coupled", "heston": "multiplicative", "heston_3_2": "multiplicative",
    "kou": "multiplicative", "merton": "multiplicative",
    "normal_inverse_gaussian": "multiplicative", "sabr": "coupled",
    "schobel_zhu": "multiplicative", "stein_stein": "multiplicative",
    "variance_gamma": "multiplicative",
    "rough_heston": "multiplicative", "quadratic_rough_heston": "multiplicative",
    "rough_bergomi": "multiplicative", "log_modulated_rough_bergomi": "multiplicative",
    "rough_stein_stein": "multiplicative", "rough_sabr": "coupled",
}
PRICE_DELTA_BINDING_SPECS = tuple(
    PriceDeltaBindingSpec(binding,
        "closed_form_bump" if binding.engine == "equity_closed_form" else
        PRICE_DELTA_PATH_STRATEGIES[binding.model])
    for binding in PRODUCT_BINDING_SPECS
    if binding.asset_class == "equity" and binding.engine in {
        "equity_markovian", "equity_closed_form", "equity_lsm_exact", "equity_lsm_fixed", "equity_n_factor", "equity_volterra_fft"}
)
GENERATED_CLOSED_FORM_POLICY_PATHS = tuple(
    f"{binding.unit_path}_impl.cuh" for binding in PRODUCT_BINDING_SPECS
    if binding.engine == "equity_closed_form" and binding.product != "european_option"
)
GENERATED_PRICE_DELTA_BINDING_PATHS = tuple(
    path for binding in PRICE_DELTA_BINDING_SPECS for path in binding.paths
)

# Sensitivities inherit the exact input/variant contract of an existing price
# recipe. They do not reserve new random streams: CRN aliases are explicit.
PRICE_DELTA_SOURCE_DATASETS = {
    dataset.recipe_path: dataset for dataset in AVAILABLE_DATASET_SPECS
    if dataset.dataset_kind == "prices" and any(
        (spec.pricing.model, spec.pricing.product, spec.pricing.engine)
        == (dataset.model, dataset.product, dataset.engine)
        for spec in PRICE_DELTA_BINDING_SPECS
    )
}
PRICE_DELTA_DATASET_SPECS = tuple(
    replace(dataset, dataset_kind="price_delta", owner="generated",
            dataset_id=dataset.dataset_id + "_price_delta",
            recipe_path=dataset.recipe_path.replace("/prices/", "/price_delta/")
                .replace("/" + dataset.dataset_id + "/", "/" + dataset.dataset_id + "_price_delta/"),
            template=("catalog/pricing/price_delta/prepared_generator.cpp.tpl"
                      if dataset.engine == "equity_n_factor" else
                      "catalog/pricing/price_delta/volterra_generator.cpp.tpl"
                      if dataset.engine == "equity_volterra_fft" else
                      "catalog/pricing/price_delta/generator.cpp.tpl"),
            layout="aligned_price_delta_rows", numerical_profile="price_delta_production_paths")
    for dataset in PRICE_DELTA_SOURCE_DATASETS.values()
)
PRICE_DELTA_SOURCE_BY_RECIPE = {
    delta.recipe_path: source for delta, source in zip(
        PRICE_DELTA_DATASET_SPECS, PRICE_DELTA_SOURCE_DATASETS.values(), strict=True)
}
DATASET_SPECS += PRICE_DELTA_DATASET_SPECS
AVAILABLE_DATASET_SPECS += PRICE_DELTA_DATASET_SPECS
RNG_COMMON_RANDOM_NUMBER_ALLOWLIST = frozenset(
    tuple(sorted((delta.recipe_path, source.recipe_path)))
    for delta in PRICE_DELTA_DATASET_SPECS
    if (source := PRICE_DELTA_SOURCE_BY_RECIPE[delta.recipe_path]).recipe_path in RNG_DOMAIN_BY_RECIPE
)
RNG_DOMAIN_SPECS += tuple(
    replace(RNG_DOMAIN_BY_RECIPE[source.recipe_path], recipe_path=delta.recipe_path)
    for delta in PRICE_DELTA_DATASET_SPECS
    if (source := PRICE_DELTA_SOURCE_BY_RECIPE[delta.recipe_path]).recipe_path in RNG_DOMAIN_BY_RECIPE
)
RNG_DOMAIN_BY_RECIPE = {domain.recipe_path: domain for domain in RNG_DOMAIN_SPECS}
validate_rng_domain_specs(RNG_DOMAIN_SPECS, RNG_COMMON_RANDOM_NUMBER_ALLOWLIST)
for _dataset in PRICE_DELTA_DATASET_SPECS:
    validate_dataset_spec(_dataset)


def pricing_launch_family(binding: ProductBindingSpec) -> str:
    """Classify execution, without duplicating any measured launch numbers."""
    if binding.engine == "fixed_income_closed_form":
        return "jamshidian" if binding.product == "european_swaption" else "closed_form"
    if binding.engine == "fixed_income_lsm":
        return "terminal_forward_lsm" if binding.terminal_forward_dynamics else "gaussian_rate_lsm"
    families = {
        "equity_closed_form": "closed_form",
        "fixed_income_monte_carlo": "fixed_income_mc",
        "equity_lsm_fixed": "equity_lsm",
        "equity_lsm_exact": "equity_lsm",
        "equity_n_factor": "rough_n_factor",
        "equity_volterra_fft": "rough_fft",
    }
    if binding.engine in families:
        return families[binding.engine]
    if isinstance(binding.manifest_binding, Binding):
        return "equity_exact_mc" if binding.manifest_binding.time_kind == "exact" else "equity_step_mc"
    raise ValueError(f"Unclassified pricing launch family: {binding.unit_path}")


validate_price_capability_graph(DATASET_SPECS, PRODUCT_BINDING_SPECS)
GENERATED_PRODUCT_BINDING_SPECS = tuple(
    spec for spec in PRODUCT_BINDING_SPECS if spec.owner == "generated"
)
DECLARED_PRODUCT_BINDING_PATHS = tuple(
    path for spec in PRODUCT_BINDING_SPECS for path in spec.paths
)
GENERATED_PRODUCT_BINDING_PATHS = tuple(
    path for spec in GENERATED_PRODUCT_BINDING_SPECS for path in spec.paths
)


CAPABILITY_EXCEPTIONS = (
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
