"""Uniform publication pipeline for stochastic one-asset equity models.

Every model/product module is intentionally a two-line CLI wrapper.  This file
owns the invariant hierarchy (Premia, QuantLib specialized, QuantLib Monte
Carlo), row-local fallback, tolerances, report creation, and catalogue paths.
Model modules provide only a declarative :class:`EquityValidationSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from validation.dataset_validation import (
    ValidationPolicy,
    build_validation_section,
    premia_row_exception,
    quantlib_row_exception,
    run_dataset_validation_cli,
    unavailable_engine,
)
from validation.hierarchy import (
    BackendBatchResult,
    ValidationEngine,
    isolate_backend_exceptions,
)
from validation.premia.model.equity.terminal_option import (
    validation_from_premia_terminal_option,
)
from validation.premia.model.equity.path_option import (
    validation_batch_from_premia_path_option,
)
from validation.quantlib.price_validation import (
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
)
from validation.reporting import (
    DatasetValidationReport,
    validation_fingerprint,
)


QuantLibValidator = Callable[
    [
        str | Path,
        str,
        ValidationTolerances,
        ValidationRegime,
        Sequence[str] | None,
    ],
    PriceValidationReport,
]
PremiaValidator = Callable[
    [
        str | Path,
        str,
        ValidationTolerances,
        ValidationRegime,
        Sequence[str] | None,
    ],
    PriceValidationReport,
]


COMMON_PRODUCT_KINDS = frozenset(
    {
        "asian_call",
        "asian_put",
        "asset_or_nothing_call",
        "asset_or_nothing_put",
        "athena_autocall",
        "cliquet",
        "digital_call",
        "digital_put",
        "double_knock_out_call",
        "double_knock_out_put",
        "down_and_in_put",
        "down_and_out_put",
        "european_call",
        "european_put",
        "forward_start_call",
        "forward_start_put",
        "gap_call",
        "gap_put",
        "geometric_asian_call",
        "geometric_asian_put",
        "lookback_option",
        "phoenix_autocall",
        "phoenix_memory_autocall",
        "range_accrual",
        "straddle",
        "up_and_in_call",
        "up_and_out_call",
        "up_no_touch",
        "up_one_touch",
    }
)

TERMINAL_PRODUCT_KINDS = frozenset(
    {
        "asset_or_nothing_call",
        "asset_or_nothing_put",
        "digital_call",
        "digital_put",
        "european_call",
        "european_put",
        "gap_call",
        "gap_put",
        "straddle",
    }
)

_PATH_PRODUCT_KINDS = COMMON_PRODUCT_KINDS - TERMINAL_PRODUCT_KINDS

_PRODUCT_FOLDERS = {
    "lookback_option": "lookback_options",
    "up_no_touch": "up_no_touches",
    "up_one_touch": "up_one_touches",
}


@dataclass(frozen=True)
class PremiaMethodCandidate:
    """One compatible Premia method, ordered before lower-priority fallbacks."""

    pricing_method: str
    vanilla_method: str | None = None
    digital_method: str | None = None
    path_method: str | None = None
    asian_put_via_call_parity: bool = True
    validator: PremiaValidator | None = None


@dataclass(frozen=True)
class EquityValidationSpec:
    """Declarative independent-pricer coverage for one equity model."""

    model_name: str
    product_kinds: frozenset[str]
    premia_products: frozenset[str]
    premia_methods: dict[str, str]
    premia_method_candidates: dict[
        str, tuple[PremiaMethodCandidate, ...]
    ] | None = None
    premia_path_products: frozenset[str] = frozenset()
    bias_explanations: dict[str, str] | None = None
    quantlib_specialized_products: frozenset[str] = frozenset()
    quantlib_specialized_methods: dict[str, str] | None = None
    quantlib_monte_carlo_products: frozenset[str] = frozenset()
    quantlib_monte_carlo_methods: dict[str, str] | None = None
    quantlib_validator: QuantLibValidator | None = None
    quantlib_specialized_unavailable_reason: str = (
        "no compatible QuantLib specialized pricer matches this model/product contract"
    )
    quantlib_monte_carlo_unavailable_reason: str = (
        "no compatible QuantLib Monte Carlo matches this model/product contract"
    )
    enforce_directional_bias: bool = False
    enforce_statistical_bias: bool = True
    near_zero_relative_materiality: float | None = None

    def __post_init__(self) -> None:
        unknown = (
            self.premia_products
            | self.quantlib_specialized_products
            | self.quantlib_monte_carlo_products
        ) - self.product_kinds
        if unknown:
            raise ValueError(
                f"{self.model_name}: engine coverage contains unknown products: "
                + ", ".join(sorted(unknown))
            )
        if not self.premia_path_products <= self.premia_products:
            raise ValueError(
                f"{self.model_name}: Premia path products must be covered products."
            )
        candidate_products = set((self.premia_method_candidates or {}).keys())
        if not candidate_products <= self.premia_products:
            raise ValueError(
                f"{self.model_name}: Premia candidates contain uncovered products: "
                + ", ".join(sorted(candidate_products - self.premia_products))
            )
        empty_candidates = tuple(
            product
            for product, candidates in (
                self.premia_method_candidates or {}
            ).items()
            if not candidates
        )
        if empty_candidates:
            raise ValueError(
                f"{self.model_name}: empty Premia candidate lists for "
                + ", ".join(sorted(empty_candidates))
            )
        for products, methods, label in (
            (self.premia_products, self.premia_methods, "Premia"),
            (
                self.quantlib_specialized_products,
                self.quantlib_specialized_methods or {},
                "QuantLib specialized",
            ),
            (
                self.quantlib_monte_carlo_products,
                self.quantlib_monte_carlo_methods or {},
                "QuantLib Monte Carlo",
            ),
        ):
            missing = products - methods.keys()
            if missing:
                raise ValueError(
                    f"{self.model_name}: {label} methods missing for "
                    + ", ".join(sorted(missing))
                )


def product_folder(product_kind: str) -> str:
    """Return the canonical plural catalogue folder for one product kind."""

    return _PRODUCT_FOLDERS.get(product_kind, product_kind + "s")


def _validation_tolerances(product_kind: str) -> ValidationTolerances:
    """Preserve the established tolerances by numerical payoff family."""

    if product_kind in {
        "up_and_out_call",
        "up_and_in_call",
        "down_and_out_put",
        "down_and_in_put",
        "double_knock_out_call",
        "double_knock_out_put",
        "lookback_option",
    }:
        return ValidationTolerances(absolute=2.0e-3, relative=1.5e-1)
    if product_kind in {"up_one_touch", "up_no_touch"}:
        return ValidationTolerances(absolute=3.0e-3, relative=1.5e-1)
    if product_kind in {
        "athena_autocall",
        "phoenix_autocall",
        "phoenix_memory_autocall",
        "cliquet",
        "range_accrual",
    }:
        return ValidationTolerances(
            absolute=5.0e-4,
            relative=2.0e-3,
            standard_error_multiplier=5.0,
            bias_standard_errors=5.0,
        )
    if product_kind in {
        "asian_call",
        "asian_put",
        "forward_start_call",
        "forward_start_put",
        "geometric_asian_call",
        "geometric_asian_put",
    }:
        return ValidationTolerances(
            absolute=2.0e-5,
            relative=2.0e-3,
            standard_error_multiplier=5.0,
            bias_standard_errors=5.0,
        )
    if product_kind in {
        "asset_or_nothing_call",
        "asset_or_nothing_put",
        "gap_call",
        "gap_put",
    }:
        return ValidationTolerances(absolute=3.0e-4, relative=2.0e-3)
    if product_kind in {"digital_call", "digital_put", "american_call", "american_put"}:
        return ValidationTolerances(absolute=2.0e-4, relative=2.0e-3)
    return ValidationTolerances()


def _tolerance_description(product_kind: str) -> str:
    tolerances = _validation_tolerances(product_kind)
    return (
        f"{tolerances.absolute:g} absolute + {tolerances.relative:g} relative "
        f"+ {tolerances.standard_error_multiplier:g} combined standard errors"
    )


def _premia_engines(
    spec: EquityValidationSpec,
    product_kind: str,
) -> tuple[ValidationEngine, ...]:
    if product_kind not in spec.premia_products:
        return (
            unavailable_engine(
                "Premia",
                "specialized pricer",
                "no compatible Premia engine matches this model/product contract",
            ),
        )

    candidates = (spec.premia_method_candidates or {}).get(
        product_kind,
        (PremiaMethodCandidate(spec.premia_methods[product_kind]),),
    )

    def engine(candidate: PremiaMethodCandidate) -> ValidationEngine:
        def validate(
            price_dataset_path: Path,
            regime: ValidationRegime,
            row_ids: tuple[str, ...],
        ) -> BackendBatchResult:
            tolerances = _validation_tolerances(product_kind)
            if candidate.validator is not None:
                return isolate_backend_exceptions(
                    row_ids,
                    lambda selected: candidate.validator(
                        price_dataset_path,
                        product_kind,
                        tolerances,
                        regime,
                        selected,
                    ),
                    premia_row_exception,
                )
            if product_kind in spec.premia_path_products:
                return validation_batch_from_premia_path_option(
                    price_dataset_path,
                    spec.model_name,
                    product_kind,
                    tolerances=tolerances,
                    regime=regime,
                    row_ids=row_ids,
                    method=candidate.path_method,
                    asian_put_via_call_parity=(
                        candidate.asian_put_via_call_parity
                    ),
                )

            return isolate_backend_exceptions(
                row_ids,
                lambda selected: validation_from_premia_terminal_option(
                    price_dataset_path,
                    spec.model_name,
                    product_kind,
                    tolerances=tolerances,
                    vanilla_method=candidate.vanilla_method,
                    digital_method=candidate.digital_method,
                    regime=regime,
                    row_ids=selected,
                ),
                premia_row_exception,
            )

        return ValidationEngine(
            "Premia",
            "specialized pricer",
            validate,
            pricing_method=candidate.pricing_method,
        )

    return tuple(engine(candidate) for candidate in candidates)


def _quantlib_engine(
    spec: EquityValidationSpec,
    product_kind: str,
    monte_carlo: bool,
) -> ValidationEngine:
    products = (
        spec.quantlib_monte_carlo_products
        if monte_carlo
        else spec.quantlib_specialized_products
    )
    method = "Monte Carlo" if monte_carlo else "specialized pricer"
    if product_kind not in products or spec.quantlib_validator is None:
        return unavailable_engine(
            "QuantLib",
            method,
            (
                spec.quantlib_monte_carlo_unavailable_reason
                if monte_carlo
                else spec.quantlib_specialized_unavailable_reason
            ),
        )

    def validate(
        price_dataset_path: Path,
        regime: ValidationRegime,
        row_ids: tuple[str, ...],
    ) -> BackendBatchResult:
        return isolate_backend_exceptions(
            row_ids,
            lambda selected: spec.quantlib_validator(
                price_dataset_path,
                product_kind,
                _validation_tolerances(product_kind),
                regime,
                selected,
            ),
            quantlib_row_exception,
        )

    methods = (
        spec.quantlib_monte_carlo_methods
        if monte_carlo
        else spec.quantlib_specialized_methods
    ) or {}
    return ValidationEngine(
        "QuantLib",
        method,
        validate,
        pricing_method=methods[product_kind],
    )


def engine_plan(
    spec: EquityValidationSpec,
    product_kind: str,
) -> tuple[ValidationEngine, ...]:
    """Build Premia candidates followed by both QuantLib hierarchy slots."""

    if product_kind not in spec.product_kinds:
        raise ValueError(
            f"Unsupported {spec.model_name} product '{product_kind}'."
        )
    return (
        *_premia_engines(spec, product_kind),
        _quantlib_engine(spec, product_kind, monte_carlo=False),
        _quantlib_engine(spec, product_kind, monte_carlo=True),
    )


def validate_dataset(
    price_dataset_path: str | Path,
    spec: EquityValidationSpec,
    product_kind: str,
) -> DatasetValidationReport:
    """Run one model/product hierarchy and return its persistent report."""

    path = Path(price_dataset_path).resolve()
    policy = ValidationPolicy(
        tolerance=_tolerance_description(product_kind),
        bias_explanation=(spec.bias_explanations or {}).get(product_kind),
        enforce_directional_bias=(
            spec.enforce_directional_bias
            or product_kind in spec.quantlib_monte_carlo_products
        ),
        enforce_statistical_bias=spec.enforce_statistical_bias,
        near_zero_relative_materiality=(
            spec.near_zero_relative_materiality
        ),
    )
    return DatasetValidationReport(
        validation_fingerprint=validation_fingerprint(path),
        core=build_validation_section(
            path, "core", engine_plan(spec, product_kind), policy
        ),
        stress=build_validation_section(
            path, "stress", engine_plan(spec, product_kind), policy
        ),
    )


def run_product_validation_cli(
    spec: EquityValidationSpec,
    product_kind: str,
) -> int:
    """Expose the identical two-path CLI used by every product module."""

    return run_dataset_validation_cli(
        lambda path: validate_dataset(path, spec, product_kind),
        f"Validate one {spec.model_name} {product_kind} dataset.",
    )


__all__ = (
    "COMMON_PRODUCT_KINDS",
    "EquityValidationSpec",
    "PremiaMethodCandidate",
    "TERMINAL_PRODUCT_KINDS",
    "engine_plan",
    "product_folder",
    "run_product_validation_cli",
    "validate_dataset",
)
