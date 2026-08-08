"""Unified independent validation for every Black-Scholes price dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
from validation.premia.model.equity.black_scholes.path_option import (
    validation_from_premia_black_scholes_path_option,
)
from validation.premia.model.equity.black_scholes.specialized_option import (
    SUPPORTED_PRODUCTS as PREMIA_SPECIALIZED_PRODUCTS,
    validation_from_premia_black_scholes_specialized_option,
)
from validation.premia.model.equity.terminal_option import (
    validation_from_premia_terminal_option,
)
from validation.quantlib.model.equity.black_scholes.monte_carlo_option import (
    SUPPORTED_PRODUCTS as QUANTLIB_MONTE_CARLO_PRODUCTS,
    validation_from_quantlib_black_scholes_monte_carlo_option,
)
from validation.quantlib.model.equity.black_scholes.specialized_option import (
    SUPPORTED_PRODUCTS as QUANTLIB_SPECIALIZED_PRODUCTS,
    validation_from_quantlib_black_scholes_specialized_option,
)
from validation.quantlib.price_validation import (
    ValidationRegime,
)
from validation.reporting import (
    DatasetValidationReport,
    ValidationDisplayReport,
    validation_fingerprint,
)


PremiaKind = Literal["terminal", "path", "specialized"]


@dataclass(frozen=True)
class ProductValidationSpec:
    """Compatible references and comparison semantics for one product."""

    product_kind: str
    premia_kind: PremiaKind | None = None
    tolerance: str = (
        "5e-7 absolute + 5e-5 relative + 5 combined standard errors"
    )
    bias_explanation: str | None = None
    enforce_directional_bias: bool = False


_TERMINAL_PRODUCTS = frozenset(
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

_PATH_BOUNDS = {
    "up_and_out_call": (
        "expected; discrete monitoring knocks out fewer paths than continuous monitoring"
    ),
    "up_and_in_call": (
        "expected; discrete monitoring activates fewer paths than continuous monitoring"
    ),
    "down_and_out_put": (
        "expected; discrete monitoring knocks out fewer paths than continuous monitoring"
    ),
    "down_and_in_put": (
        "expected; discrete monitoring activates fewer paths than continuous monitoring"
    ),
    "double_knock_out_call": (
        "expected; discrete monitoring knocks out fewer paths than continuous monitoring"
    ),
    "double_knock_out_put": (
        "expected; discrete monitoring knocks out fewer paths than continuous monitoring"
    ),
    "lookback_option": "expected; a discrete maximum cannot exceed the continuous maximum",
}

_SPECIALIZED_PATH_BIAS = {
    "asian_call": "expected; CUDA averages the daily grid while Premia integrates continuously",
    "asian_put": "expected; CUDA averages the daily grid while Premia integrates continuously",
    "up_no_touch": (
        "expected; discrete monitoring misses some barrier hits and raises a no-touch price"
    ),
    "up_one_touch": (
        "expected; discrete monitoring misses some barrier hits and lowers a one-touch price"
    ),
}

_PRODUCT_KINDS = frozenset(
    _TERMINAL_PRODUCTS
    | _PATH_BOUNDS.keys()
    | QUANTLIB_SPECIALIZED_PRODUCTS
    | QUANTLIB_MONTE_CARLO_PRODUCTS
)


def _spec(product_kind: str) -> ProductValidationSpec:
    if product_kind not in _PRODUCT_KINDS:
        raise ValueError(f"Unsupported Black-Scholes product '{product_kind}'.")
    if product_kind in _PATH_BOUNDS:
        return ProductValidationSpec(
            product_kind=product_kind,
            premia_kind="path",
            tolerance=(
                "1e-5 absolute + 5e-5 relative + 5 combined standard errors"
            ),
            bias_explanation=_PATH_BOUNDS[product_kind],
        )
    if product_kind in _TERMINAL_PRODUCTS:
        return ProductValidationSpec(product_kind, premia_kind="terminal")
    if product_kind in PREMIA_SPECIALIZED_PRODUCTS:
        if product_kind in _SPECIALIZED_PATH_BIAS:
            return ProductValidationSpec(
                product_kind,
                premia_kind="specialized",
                tolerance=(
                    "exact Black-Scholes L2 discretization bound + 1e-5 absolute "
                    "+ 5e-5 relative + 5 combined standard errors"
                    if product_kind in {"asian_call", "asian_put"}
                    else "1e-5 absolute + 5e-5 relative + 5 combined standard errors"
                ),
                bias_explanation=_SPECIALIZED_PATH_BIAS[product_kind],
            )
        return ProductValidationSpec(product_kind, premia_kind="specialized")
    return ProductValidationSpec(
        product_kind,
        enforce_directional_bias=(
            product_kind in QUANTLIB_MONTE_CARLO_PRODUCTS
        ),
    )


def _premia_terminal_engine(
    spec: ProductValidationSpec,
) -> ValidationEngine:
    def validate(
        price_dataset_path: Path,
        regime: ValidationRegime,
        row_ids: tuple[str, ...],
    ) -> BackendBatchResult:
        return isolate_backend_exceptions(
            row_ids,
            lambda selected: validation_from_premia_terminal_option(
                price_dataset_path,
                "black_scholes",
                spec.product_kind,
                regime=regime,
                row_ids=selected,
            ),
            premia_row_exception,
        )

    return ValidationEngine("Premia", "specialized pricer", validate)


def _premia_path_engine(spec: ProductValidationSpec) -> ValidationEngine:
    def validate(
        price_dataset_path: Path,
        regime: ValidationRegime,
        row_ids: tuple[str, ...],
    ) -> BackendBatchResult:
        return isolate_backend_exceptions(
            row_ids,
            lambda selected: validation_from_premia_black_scholes_path_option(
                price_dataset_path,
                spec.product_kind,
                regime=regime,
                row_ids=selected,
            ),
            premia_row_exception,
        )

    return ValidationEngine("Premia", "specialized pricer", validate)


def _premia_specialized_engine(spec: ProductValidationSpec) -> ValidationEngine:
    def validate(
        price_dataset_path: Path,
        regime: ValidationRegime,
        row_ids: tuple[str, ...],
    ) -> BackendBatchResult:
        return isolate_backend_exceptions(
            row_ids,
            lambda selected: (
                validation_from_premia_black_scholes_specialized_option(
                    price_dataset_path,
                    spec.product_kind,
                    regime=regime,
                    row_ids=selected,
                )
            ),
            premia_row_exception,
        )

    return ValidationEngine("Premia", "specialized pricer", validate)


def _unavailable(reference: str, method: str, reason: str) -> ValidationEngine:
    return unavailable_engine(reference, method, reason)


def _engine_plan(
    spec: ProductValidationSpec,
    path_reference_pairs: int = 1024,
) -> tuple[ValidationEngine, ...]:
    if spec.premia_kind == "terminal":
        premia = _premia_terminal_engine(spec)
    elif spec.premia_kind == "path":
        premia = _premia_path_engine(spec)
    elif spec.premia_kind == "specialized":
        premia = _premia_specialized_engine(spec)
    else:
        premia = _unavailable(
            "Premia",
            "specialized pricer",
            "no compatible specialized Premia contract is exposed",
        )

    if spec.product_kind in QUANTLIB_SPECIALIZED_PRODUCTS:
        def quantlib_specialized(
            price_dataset_path: Path,
            regime: ValidationRegime,
            row_ids: tuple[str, ...],
        ) -> BackendBatchResult:
            return isolate_backend_exceptions(
                row_ids,
                lambda selected: (
                    validation_from_quantlib_black_scholes_specialized_option(
                        price_dataset_path,
                        spec.product_kind,
                        regime=regime,
                        row_ids=selected,
                    )
                ),
                quantlib_row_exception,
            )

        specialized = ValidationEngine(
            "QuantLib", "specialized pricer", quantlib_specialized
        )
    else:
        specialized = _unavailable(
            "QuantLib",
            "specialized pricer",
            "no specialized engine matches the catalogue contract",
        )

    if spec.product_kind in QUANTLIB_MONTE_CARLO_PRODUCTS:
        def quantlib_monte_carlo(
            price_dataset_path: Path,
            regime: ValidationRegime,
            row_ids: tuple[str, ...],
        ) -> BackendBatchResult:
            return isolate_backend_exceptions(
                row_ids,
                lambda selected: (
                    validation_from_quantlib_black_scholes_monte_carlo_option(
                        price_dataset_path,
                        spec.product_kind,
                        regime=regime,
                        row_ids=selected,
                        path_reference_pairs=path_reference_pairs,
                    )
                ),
                quantlib_row_exception,
            )

        monte_carlo = ValidationEngine(
            "QuantLib", "Monte Carlo", quantlib_monte_carlo
        )
    else:
        monte_carlo = _unavailable(
            "QuantLib",
            "Monte Carlo",
            "no independent QuantLib Monte Carlo contract is required or available",
        )
    return premia, specialized, monte_carlo


def _display_report(
    price_dataset_path: Path,
    regime: ValidationRegime,
    spec: ProductValidationSpec,
) -> ValidationDisplayReport:
    policy = ValidationPolicy(
        tolerance=spec.tolerance,
        bias_explanation=spec.bias_explanation,
        enforce_directional_bias=spec.enforce_directional_bias,
    )
    report = build_validation_section(
        price_dataset_path, regime, _engine_plan(spec), policy
    )
    if (
        spec.product_kind in {"up_no_touch", "up_one_touch"}
        and report.systematic_bias
    ):
        # A 100-row stress regime can cross the 60% sign alarm by sampling
        # noise.  Confirm such an alarm with four times as many independent
        # antithetic reference pairs before accepting or rejecting it.
        report = build_validation_section(
            price_dataset_path,
            regime,
            _engine_plan(spec, path_reference_pairs=4096),
            policy,
        )
    return report


def validate_dataset(
    price_dataset_path: str | Path,
    product_kind: str,
) -> DatasetValidationReport:
    """Run the ordered hierarchy once and return its persistent report."""

    path = Path(price_dataset_path).resolve()
    spec = _spec(product_kind)
    return DatasetValidationReport(
        validation_fingerprint=validation_fingerprint(path),
        core=_display_report(path, "core", spec),
        stress=_display_report(path, "stress", spec),
    )


def run_product_validation_cli(product_kind: str) -> int:
    """Provide the identical two-path CLI used by every thin product module."""

    return run_dataset_validation_cli(
        lambda path: validate_dataset(path, product_kind),
        f"Validate one Black-Scholes {product_kind} dataset.",
    )
