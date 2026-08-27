"""Independent QuantLib engines for co-terminal Bermudan swaptions."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import QuantLib as ql

from validation.quantlib.parameters import finite_number, positive_number
from validation.quantlib.price_validation import (
    PriceResultRow,
    PriceComparison,
    PriceValidationReport,
    ValidationRegime,
    ValidationTolerances,
    load_parameter_rows,
    load_price_validation_input,
    select_validation_regime,
    select_validation_row_ids,
    summarize_price_comparisons,
    validation_from_reference,
)
from validation.quantlib.term_structure import (
    BUSINESS_DAYS_PER_YEAR,
    DAY_COUNTER,
    REFERENCE_DATE,
    date_from_time,
)
from validation.quantlib.rate_option import RateModelFactory
from validation.quantlib.swaption import swaption_price


EngineKind = Literal["tree", "fd_hull_white", "fd_g2"]


@dataclass(frozen=True)
class PreparedBermudanModel:
    """One QuantLib short-rate model and the curve seen by its swap."""

    model: Any
    term_structure: ql.YieldTermStructureHandle
    engine_kind: EngineKind


@dataclass(frozen=True)
class BermudanEngineConfiguration:
    """Deterministic discretization used by one QuantLib reference run."""

    tree_steps: int = 300
    time_grid: int = 150
    x_grid: int = 150
    y_grid: int = 75
    damping_steps: int = 0

    def __post_init__(self) -> None:
        values = (self.tree_steps, self.time_grid, self.x_grid, self.y_grid)
        if any(value <= 0 for value in values) or self.damping_steps < 0:
            raise ValueError("QuantLib engine grid sizes must be positive.")


BermudanModelFactory = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any]],
    PreparedBermudanModel,
]


def short_exercise_engine(
    product: Mapping[str, Any],
    short_engine: EngineKind,
    regular_engine: EngineKind = "tree",
) -> EngineKind:
    """Avoid QuantLib's tree loss of the first very-short exercise date."""

    first_exercise = positive_number(
        product, "first_exercise_time", "Bermudan swaption"
    )
    return short_engine if first_exercise <= 0.25 else regular_engine


def _positive_integer(
    parameters: Mapping[str, Any], field: str, context: str
) -> int:
    value = positive_number(parameters, field, context)
    integer = round(value)
    if not math.isclose(value, integer, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{context}: {field} must be an integer.")
    return integer


def bermudan_swaption_times(product: Mapping[str, Any]) -> tuple[float, ...]:
    """Return the swap start and every co-terminal payment time."""

    context = "Bermudan swaption"
    first_exercise = positive_number(product, "first_exercise_time", context)
    payment_interval = positive_number(product, "payment_interval", context)
    payment_count = _positive_integer(product, "payment_count", context)
    return tuple(
        first_exercise + index * payment_interval
        for index in range(payment_count + 1)
    )


def _exercise_times(product: Mapping[str, Any]) -> tuple[float, ...]:
    context = "Bermudan swaption"
    first_exercise = positive_number(product, "first_exercise_time", context)
    payment_interval = positive_number(product, "payment_interval", context)
    payment_count = _positive_integer(product, "payment_count", context)
    exercise_count = _positive_integer(product, "exercise_count", context)
    if exercise_count > payment_count:
        raise ValueError(
            f"{context}: exercise_count must not exceed payment_count."
        )
    return tuple(
        first_exercise + index * payment_interval
        for index in range(exercise_count)
    )


def _swaption(
    term_structure: ql.YieldTermStructureHandle,
    product: Mapping[str, Any],
    side: str,
) -> ql.Swaption:
    """Build the physical-settlement swaption matching the CUDA contract."""

    if side not in {"payer", "receiver"}:
        raise ValueError(f"Unsupported Bermudan swaption side '{side}'.")
    context = "Bermudan swaption"
    notional = positive_number(product, "notional", context)
    strike = finite_number(product, "strike", context)
    if strike < 0.0:
        raise ValueError(f"{context}: strike must be non-negative.")
    accrual = positive_number(product, "accrual_fraction", context)
    payment_interval = positive_number(product, "payment_interval", context)
    if not math.isclose(accrual, payment_interval, rel_tol=0.0, abs_tol=5.0e-8):
        raise ValueError(
            f"{context}: this QuantLib adapter requires regular accruals equal "
            "to the payment interval."
        )

    payment_times = bermudan_swaption_times(product)
    payment_dates = [date_from_time(time) for time in payment_times]
    schedule = ql.Schedule(
        payment_dates, ql.NullCalendar(), ql.Unadjusted
    )
    interval_days = round(payment_interval * BUSINESS_DAYS_PER_YEAR)
    index = ql.IborIndex(
        "AI Factory synthetic",
        ql.Period(interval_days, ql.Days),
        0,
        ql.USDCurrency(),
        ql.NullCalendar(),
        ql.Unadjusted,
        False,
        DAY_COUNTER,
        term_structure,
    )
    swap_type = (
        ql.VanillaSwap.Payer if side == "payer" else ql.VanillaSwap.Receiver
    )
    swap = ql.VanillaSwap(
        swap_type,
        notional,
        schedule,
        strike,
        DAY_COUNTER,
        schedule,
        index,
        0.0,
        DAY_COUNTER,
    )
    exercise = ql.BermudanExercise(
        [date_from_time(time) for time in _exercise_times(product)]
    )
    return ql.Swaption(swap, exercise)


def _engine(
    prepared: PreparedBermudanModel,
    configuration: BermudanEngineConfiguration,
) -> ql.PricingEngine:
    if prepared.engine_kind == "tree":
        return ql.TreeSwaptionEngine(
            prepared.model,
            configuration.tree_steps,
            prepared.term_structure,
        )
    if prepared.engine_kind == "fd_hull_white":
        return ql.FdHullWhiteSwaptionEngine(
            prepared.model,
            configuration.time_grid,
            configuration.x_grid,
            configuration.damping_steps,
        )
    if prepared.engine_kind == "fd_g2":
        return ql.FdG2SwaptionEngine(
            prepared.model,
            configuration.time_grid,
            configuration.x_grid,
            configuration.y_grid,
            configuration.damping_steps,
        )
    raise ValueError(f"Unsupported QuantLib engine '{prepared.engine_kind}'.")


def bermudan_swaption_price(
    prepared: PreparedBermudanModel,
    product: Mapping[str, Any],
    side: str,
    configuration: BermudanEngineConfiguration = BermudanEngineConfiguration(),
) -> float:
    """Price one contract with the selected independent QuantLib engine."""

    ql.Settings.instance().evaluationDate = REFERENCE_DATE
    instrument = _swaption(prepared.term_structure, product, side)
    instrument.setPricingEngine(_engine(prepared, configuration))
    return max(float(instrument.NPV()), 0.0)


def validation_from_quantlib_bermudan_swaption(
    price_dataset_path: str | Path,
    model_factory: BermudanModelFactory,
    side: str,
    require_curve: bool,
    configuration: BermudanEngineConfiguration = BermudanEngineConfiguration(),
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Compare a selected price dataset batch with QuantLib."""

    def reference(
        model: Mapping[str, Any],
        curve: Mapping[str, Any] | None,
        product: Mapping[str, Any],
        _: PriceResultRow,
    ) -> float:
        return bermudan_swaption_price(
            model_factory(model, curve, product),
            product,
            side,
            configuration,
        )

    return validation_from_reference(
        price_dataset_path,
        reference,
        tolerances,
        require_curve,
        regime,
        row_ids,
    )


def european_exercise_lower_bound(
    model: Mapping[str, Any],
    curve: Mapping[str, Any] | None,
    product: Mapping[str, Any],
    model_factory: RateModelFactory,
    side: str,
) -> float:
    """Return the largest analytical European exercise opportunity."""

    context = "Bermudan swaption"
    notional = positive_number(product, "notional", context)
    strike = finite_number(product, "strike", context)
    accrual = positive_number(product, "accrual_fraction", context)
    first_exercise = positive_number(product, "first_exercise_time", context)
    interval = positive_number(product, "payment_interval", context)
    payment_count = _positive_integer(product, "payment_count", context)
    exercise_count = _positive_integer(product, "exercise_count", context)
    prices = []
    for exercise in range(exercise_count):
        european_product = {
            "notional": notional,
            "strike": strike,
            "accrual_fraction": accrual,
            "exercise_time": first_exercise + exercise * interval,
            "payment_interval": interval,
            "payment_count": payment_count - exercise,
        }
        prices.append(
            swaption_price(
                model_factory(model, curve, european_product),
                european_product,
                side,
            )
        )
    return max(prices)


def validation_from_quantlib_european_lower_bound(
    price_dataset_path: str | Path,
    model_factory: RateModelFactory,
    side: str,
    require_curve: bool,
    tolerances: ValidationTolerances = ValidationTolerances(),
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Check the necessary Bermudan >= max-European inequality."""

    validation_input = select_validation_row_ids(
        select_validation_regime(
            load_price_validation_input(price_dataset_path), regime
        ),
        row_ids,
    )
    if require_curve != (validation_input.curve_dataset_path is not None):
        expected = "with" if require_curve else "without"
        raise ValueError(f"Expected a price dataset {expected} a curve reference.")
    models = load_parameter_rows(validation_input.model_dataset_path, "models")
    products = load_parameter_rows(
        validation_input.product_dataset_path, "products"
    )
    curves = (
        load_parameter_rows(validation_input.curve_dataset_path, "curves")
        if validation_input.curve_dataset_path is not None
        else None
    )
    comparisons: list[PriceComparison] = []
    for row in validation_input.rows:
        model = models[row.model_id]
        product = products[row.product_id]
        curve = curves[row.curve_id] if curves is not None else None
        comparisons.append(
            PriceComparison(
                row.row_id,
                row.generated_price,
                european_exercise_lower_bound(
                    model, curve, product, model_factory, side
                ),
                row.generated_standard_error,
                comparison_relation="generated_at_least_reference",
            )
        )
    return summarize_price_comparisons(
        validation_input.database_id, comparisons, tolerances
    )


__all__ = (
    "BermudanEngineConfiguration",
    "BermudanModelFactory",
    "PreparedBermudanModel",
    "bermudan_swaption_price",
    "bermudan_swaption_times",
    "european_exercise_lower_bound",
    "short_exercise_engine",
    "validation_from_quantlib_european_lower_bound",
    "validation_from_quantlib_bermudan_swaption",
)
