"""Unified independent validation for every Black-Scholes price dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

from validation.dataset_validation import (
    ValidationPolicy,
    build_validation_section,
    premia_row_exception,
    quantlib_row_exception,
    unavailable_engine,
)
from validation.hierarchy import (
    BackendBatchResult,
    ValidationEngine,
    isolate_backend_exceptions,
    run_validation_hierarchy,
)
from validation.model.equity.reference_pipeline import (
    REGIME_ROW_COUNTS,
    detailed_pricer,
    persist_generated_reference,
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
    load_price_validation_input,
)
from validation.reference_price_dataset import (
    ReferenceDatasetValidation,
    ReferencePrice,
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

_PREMIA_PRICING_METHODS = {
    "asset_or_nothing_call": "CF_Call + CF_Digit (static replication)",
    "asset_or_nothing_put": "CF_Put + CF_Digit (static replication)",
    "digital_call": "CF_Digit",
    "digital_put": "CF_Digit (put parity)",
    "european_call": "CF_Call",
    "european_put": "CF_Put",
    "gap_call": "CF_Call + CF_Digit (static replication)",
    "gap_put": "CF_Put + CF_Digit (static replication)",
    "straddle": "CF_Call + CF_Put",
    "up_and_out_call": "CF_CallUpOut (continuous monitoring)",
    "up_and_in_call": "CF_CallUpIn (continuous monitoring)",
    "down_and_out_put": "CF_PutDownOut (continuous monitoring)",
    "down_and_in_put": "CF_PutDownIn (continuous monitoring)",
    "double_knock_out_call": (
        "CF_CallOut_KunitomoIkeda (continuous monitoring)"
    ),
    "double_knock_out_put": (
        "CF_PutOut_KunitomoIkeda (continuous monitoring)"
    ),
    "lookback_option": "CF_Fixed_CallLookBack (continuous monitoring)",
    "asian_call": "MC_FixedAsian_ExactMethod (continuous-time average)",
    "asian_put": "MC_FixedAsian_ExactMethod (continuous-time average)",
    "forward_start_call": "CF_Call (scale-invariant reduction)",
    "forward_start_put": "CF_Put (scale-invariant reduction)",
    "geometric_asian_call": "CF_Call (lognormal reduction)",
    "geometric_asian_put": "CF_Put (lognormal reduction)",
    "range_accrual": "CF_Digit (marginal replication)",
    "up_no_touch": (
        "CF_CallUpIn + CF_Call (continuous-monitoring static replication)"
    ),
    "up_one_touch": (
        "CF_CallUpIn + CF_Call (continuous-monitoring static replication)"
    ),
}

_QUANTLIB_SPECIALIZED_PRICING_METHODS = {
    "asset_or_nothing_call": "_asset_or_nothing_price (CumulativeNormalDistribution)",
    "asset_or_nothing_put": "_asset_or_nothing_price (CumulativeNormalDistribution)",
    "digital_call": "_digital_price (CumulativeNormalDistribution)",
    "digital_put": "_digital_price (CumulativeNormalDistribution)",
    "european_call": "_european_price (BlackCalculator)",
    "european_put": "_european_price (BlackCalculator)",
    "forward_start_call": "_forward_start_price (BlackCalculator)",
    "forward_start_put": "_forward_start_price (BlackCalculator)",
    "gap_call": "_gap_price (CumulativeNormalDistribution)",
    "gap_put": "_gap_price (CumulativeNormalDistribution)",
    "geometric_asian_call": "_geometric_asian_price (BlackCalculator)",
    "geometric_asian_put": "_geometric_asian_price (BlackCalculator)",
    "range_accrual": "_range_accrual_price (CumulativeNormalDistribution)",
    "straddle": "_european_price (two BlackCalculator calls)",
}

_QUANTLIB_MONTE_CARLO_PRICING_METHODS = {
    "asian_call": "MCDiscreteArithmeticAPEngine (antithetic pseudorandom)",
    "asian_put": "MCDiscreteArithmeticAPEngine (antithetic pseudorandom)",
    **{
        product: "_antithetic_path_price (GaussianPathGenerator)"
        for product in QUANTLIB_MONTE_CARLO_PRODUCTS - {"asian_call", "asian_put"}
    },
}


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

    return ValidationEngine(
        "Premia",
        "specialized pricer",
        validate,
        pricing_method=_PREMIA_PRICING_METHODS[spec.product_kind],
    )


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

    return ValidationEngine(
        "Premia",
        "specialized pricer",
        validate,
        pricing_method=_PREMIA_PRICING_METHODS[spec.product_kind],
    )


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

    return ValidationEngine(
        "Premia",
        "specialized pricer",
        validate,
        pricing_method=_PREMIA_PRICING_METHODS[spec.product_kind],
    )


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
            "QuantLib",
            "specialized pricer",
            quantlib_specialized,
            pricing_method=(
                _QUANTLIB_SPECIALIZED_PRICING_METHODS[spec.product_kind]
            ),
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
            "QuantLib",
            "Monte Carlo",
            quantlib_monte_carlo,
            pricing_method=(
                _QUANTLIB_MONTE_CARLO_PRICING_METHODS[spec.product_kind]
            ),
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
    return build_validation_section(
        price_dataset_path, regime, _engine_plan(spec), policy
    )


def live_validation_report(
    price_dataset_path: str | Path,
    product_kind: str,
) -> DatasetValidationReport:
    """Run external engines directly for diagnostic or regeneration work."""

    path = Path(price_dataset_path).resolve()
    spec = _spec(product_kind)
    return DatasetValidationReport(
        validation_fingerprint=validation_fingerprint(path),
        core=_display_report(path, "core", spec),
        stress=_display_report(path, "stress", spec),
    )


# =============================================================================
# Persistent reference generation
# =============================================================================


def _pricer_identifier(engine: ValidationEngine, product_kind: str) -> str:
    backend = engine.reference.lower().replace(" ", "_")
    method = engine.method.lower().replace(" ", "_")
    return f"{backend}_black_scholes_{product_kind}_{method}"


def _comparison_relation(
    engine: ValidationEngine,
    product_kind: str,
) -> str:
    """Describe the exact mathematical contract checked by one backend."""

    if engine.reference != "Premia":
        return "absolute"
    if product_kind in {
        "up_and_out_call",
        "down_and_out_put",
        "double_knock_out_call",
        "double_knock_out_put",
        "up_no_touch",
    }:
        return "generated_at_least_reference"
    if product_kind in {
        "up_and_in_call",
        "down_and_in_put",
        "lookback_option",
        "up_one_touch",
    }:
        return "generated_at_most_reference"
    return "absolute"


def _backend_version(engine: ValidationEngine) -> str:
    if engine.reference == "Premia":
        return "19"
    if engine.reference == "QuantLib":
        import QuantLib as ql

        return ql.__version__
    raise ValueError(f"Unsupported reference backend '{engine.reference}'.")


def _pricer_kind(engine: ValidationEngine) -> str:
    return engine.method.lower().replace(" ", "_")


def _reference_pricer_section(
    product_kind: str,
    row_count: int,
    hierarchy,
) -> dict:
    """Persist availability in order and details only for engines actually used."""

    used = {
        run.engine.label: len(run.completed_row_ids)
        for run in hierarchy.runs
        if run.completed_row_ids
    }
    section: dict = {"row_count": row_count}
    slots = (
        ("premia", hierarchy.engine_plan[0]),
        ("quantlib_specialized", hierarchy.engine_plan[1]),
        ("quantlib_monte_carlo", hierarchy.engine_plan[2]),
    )
    for slot, engine in slots:
        completed = used.get(engine.label, 0)
        if not engine.available:
            section[slot] = {"status": "not_available"}
        elif completed == 0:
            section[slot] = {"status": "available"}
        else:
            section[slot] = detailed_pricer(
                _pricer_identifier(engine, product_kind),
                engine.reference,
                _backend_version(engine),
                _pricer_kind(engine),
                engine.pricing_method or "",
                completed,
            )
    return section


def generate_reference_dataset(
    price_dataset_path: str | Path,
    reference_dataset_path: str | Path,
    product_kind: str,
) -> ReferenceDatasetValidation:
    """Run the ordered hierarchy once and persist all 1000 aligned prices."""

    path = Path(price_dataset_path).resolve()
    source = load_price_validation_input(path)
    spec = _spec(product_kind)
    started = time.perf_counter()
    prices_by_id: dict[str, ReferencePrice] = {}
    pricers: dict[str, dict] = {}
    offset = 0
    for regime, row_count in REGIME_ROW_COUNTS:
        selected_rows = source.rows[offset:offset + row_count]
        offset += row_count
        row_ids = tuple(row.row_id for row in selected_rows)
        hierarchy = run_validation_hierarchy(
            path,
            regime,
            row_ids,
            _engine_plan(spec),
        )
        if hierarchy.unresolved_row_ids:
            raise RuntimeError(
                f"{product_kind} left unresolved reference rows: "
                f"{hierarchy.unresolved_row_ids}"
            )
        source_by_id = {row.row_id: row for row in selected_rows}
        for run in hierarchy.runs:
            relation = _comparison_relation(run.engine, product_kind)
            pricer_id = _pricer_identifier(run.engine, product_kind)
            diagnostic_ids: set[str] = set()
            for report in run.reports:
                for diagnostic in report.row_diagnostics:
                    if not diagnostic.passed:
                        raise RuntimeError(
                            f"{run.engine.label} rejected {product_kind} row "
                            f"'{diagnostic.row_id}'."
                        )
                    if diagnostic.row_id in diagnostic_ids:
                        raise RuntimeError(
                            f"Duplicate diagnostic for row '{diagnostic.row_id}'."
                        )
                    diagnostic_ids.add(diagnostic.row_id)
                    row = source_by_id[diagnostic.row_id]
                    prices_by_id[row.row_id] = ReferencePrice(
                        row_id=row.row_id,
                        model_id=row.model_id,
                        product_id=row.product_id,
                        price=diagnostic.reference_price,
                        standard_error=diagnostic.reference_standard_error,
                        reference_pricer_id=pricer_id,
                        comparison_relation=relation,
                        comparison_allowance=diagnostic.allowance,
                    )
            if diagnostic_ids != set(run.completed_row_ids):
                raise RuntimeError(
                    f"{run.engine.label} diagnostics do not match completed rows."
                )
        pricers[regime] = _reference_pricer_section(
            product_kind,
            row_count,
            hierarchy,
        )
    if offset != len(source.rows) or set(prices_by_id) != {
        row.row_id for row in source.rows
    }:
        raise RuntimeError("Reference generation did not cover the source dataset.")
    prices = tuple(prices_by_id[row.row_id] for row in source.rows)
    return persist_generated_reference(
        path,
        reference_dataset_path,
        prices,
        pricers,
        time.perf_counter() - started,
        allow_systematic_bias=spec.bias_explanation is not None,
        systematic_bias_explanation=spec.bias_explanation,
    )
