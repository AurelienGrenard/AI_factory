"""Common JSON loading, error metrics, and bias checks for QuantLib prices."""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import yaml


ValidationRegime = Literal["all", "core", "stress"]
CORE_ROW_COUNT = 900
STRESS_ROW_COUNT = 100


@dataclass(frozen=True)
class ValidationTolerances:
    """Numerical thresholds shared by deterministic price validators."""

    absolute: float = 5.0e-7
    relative: float = 5.0e-5
    relative_floor: float = 1.0e-8
    # A five-sigma row gate avoids predictable family-wise false alarms when
    # thousands of independent Monte Carlo estimates are checked together.
    standard_error_multiplier: float = 5.0
    bias_standard_errors: float = 4.0

    def __post_init__(self) -> None:
        values = (
            self.absolute,
            self.relative,
            self.relative_floor,
            self.standard_error_multiplier,
            self.bias_standard_errors,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("Validation tolerances must be finite and non-negative.")


@dataclass(frozen=True)
class PriceComparison:
    """One generated price and its independent QuantLib reference."""

    row_id: str
    generated_price: float
    quantlib_price: float
    generated_standard_error: float = 0.0
    quantlib_standard_error: float = 0.0


@dataclass(frozen=True)
class PriceResultRow:
    """Source row identifiers and generated price read from one price JSON."""

    row_id: str
    model_id: str
    product_id: str
    generated_price: float
    generated_standard_error: float = 0.0
    curve_id: str | None = None


@dataclass(frozen=True)
class PriceRowDiagnostic:
    """Persist the inputs and decision of one native price comparison."""

    row_id: str
    generated_price: float
    reference_price: float
    generated_standard_error: float
    reference_standard_error: float
    allowance: float
    passed: bool


@dataclass(frozen=True)
class PriceValidationInput:
    """Resolved price rows and local source datasets required by a validator."""

    database_id: str
    model_dataset_path: Path
    product_dataset_path: Path
    rows: tuple[PriceResultRow, ...]
    curve_dataset_path: Path | None = None
    monte_carlo_paths_per_price: int | None = None


@dataclass(frozen=True)
class PriceValidationReport:
    """Uniform accuracy and signed-bias summary for one complete dataset."""

    database_id: str
    row_count: int
    passed_row_count: int
    failed_row_count: int
    failed_row_ids: tuple[str, ...]
    higher_price_count: int
    lower_price_count: int
    equal_price_count: int
    mean_signed_error: float
    mean_absolute_error: float
    root_mean_squared_error: float
    root_mean_squared_standard_error: float
    signed_error_standard_error: float
    bias_limit: float
    systematic_bias: bool
    maximum_absolute_error: float
    maximum_absolute_error_row_id: str
    maximum_relative_error: float
    maximum_relative_error_row_id: str
    row_diagnostics: tuple[PriceRowDiagnostic, ...]

    @property
    def passed(self) -> bool:
        """Return true only when every row and the aggregate bias pass."""

        return self.failed_row_count == 0 and not self.systematic_bias


def _read_json(path: Path) -> Mapping[str, Any]:
    """Read one JSON object with a path-specific diagnostic."""

    try:
        with path.open(encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read JSON '{path}': {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"JSON '{path}' must contain an object at its root.")
    return document


def _required_string(document: Mapping[str, Any], field: str, context: str) -> str:
    """Return one required non-empty JSON string."""

    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}: {field} must be a non-empty string.")
    return value


def _project_root(price_dataset_path: Path) -> Path:
    """Find the project owning the local datasets referenced by a price file."""

    starts = (price_dataset_path.parent, Path.cwd().resolve())
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "datasets").is_dir() and (
                candidate / "CMakeLists.txt"
            ).is_file():
                return candidate
    raise ValueError(
        "Could not locate the project root containing CMakeLists.txt and datasets/."
    )


def _resolve_dataset(root: Path, family: str, database_id: str) -> Path:
    """Resolve one referenced dataset by its globally unique database id."""

    search_root = root / "datasets" / family
    matches = tuple(search_root.rglob(f"{database_id}.json"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one local {family} dataset '{database_id}', "
            f"found {len(matches)} below '{search_root}'."
        )
    return matches[0]


def _catalog_monte_carlo_path_count(
    document: Mapping[str, Any], root: Path, database_id: str
) -> int | None:
    """Read the estimator sample count when the catalogue declares one."""

    catalog = document.get("catalog")
    if not isinstance(catalog, str) or not catalog:
        return None
    yaml_path = root / catalog / "dataset.yaml"
    if not yaml_path.is_file():
        return None
    try:
        yaml_document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"Cannot read catalogue YAML '{yaml_path}': {error}"
        ) from error
    summary = yaml_document.get("summary") if isinstance(yaml_document, dict) else None
    value = (
        summary.get("monte_carlo_paths_per_price")
        if isinstance(summary, dict)
        else None
    )
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(
            f"Price dataset '{database_id}': monte_carlo_paths_per_price "
            "must be a positive integer."
        )
    return value


def load_price_validation_input(
    price_dataset_path: str | Path,
) -> PriceValidationInput:
    """Validate one price envelope and resolve all referenced local datasets."""

    path = Path(price_dataset_path).expanduser().resolve()
    document = _read_json(path)
    database_id = _required_string(document, "database_id", "Price dataset")
    root = _project_root(path)

    def referenced_id(field: str) -> str:
        reference = document.get(field)
        if not isinstance(reference, dict):
            raise ValueError(f"Price dataset '{database_id}': {field} must be an object.")
        return _required_string(reference, "id", f"Price dataset '{database_id}'")

    model_path = _resolve_dataset(root, "model", referenced_id("model_dataset"))
    product_path = _resolve_dataset(
        root, "product", referenced_id("product_dataset")
    )
    curve_path = None
    if "curve_dataset" in document:
        curve_path = _resolve_dataset(
            root, "curve", referenced_id("curve_dataset")
        )

    results = document.get("results")
    row_count = document.get("row_count")
    if not isinstance(results, list) or not isinstance(row_count, int):
        raise ValueError(
            f"Price dataset '{database_id}': row_count and results are required."
        )
    if row_count <= 0 or len(results) != row_count:
        raise ValueError(
            f"Price dataset '{database_id}': row_count must match results.size()."
        )

    rows: list[PriceResultRow] = []
    row_ids: set[str] = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError(f"Price dataset '{database_id}': every result must be an object.")
        row_id = _required_string(result, "id", f"Price dataset '{database_id}'")
        if row_id in row_ids:
            raise ValueError(f"Price dataset '{database_id}': duplicate row id '{row_id}'.")
        row_ids.add(row_id)
        outputs = result.get("outputs")
        price = outputs.get("price") if isinstance(outputs, dict) else None
        if not isinstance(price, (int, float)) or not math.isfinite(float(price)):
            raise ValueError(
                f"Price dataset '{database_id}' row '{row_id}': price must be finite."
            )
        curve_id = None
        if curve_path is not None:
            curve_id = _required_string(result, "curve_id", f"Price row '{row_id}'")
        standard_error = outputs.get("standard_error", 0.0)
        if not isinstance(standard_error, (int, float)) or not math.isfinite(
            float(standard_error)
        ) or float(standard_error) < 0.0:
            raise ValueError(
                f"Price dataset '{database_id}' row '{row_id}': "
                "standard_error must be finite and non-negative."
            )
        rows.append(
            PriceResultRow(
                row_id=row_id,
                model_id=_required_string(result, "model_id", f"Price row '{row_id}'"),
                product_id=_required_string(
                    result, "product_id", f"Price row '{row_id}'"
                ),
                generated_price=float(price),
                generated_standard_error=float(standard_error),
                curve_id=curve_id,
            )
        )
    return PriceValidationInput(
        database_id=database_id,
        model_dataset_path=model_path,
        product_dataset_path=product_path,
        rows=tuple(rows),
        curve_dataset_path=curve_path,
        monte_carlo_paths_per_price=_catalog_monte_carlo_path_count(
            document, root, database_id
        ),
    )


def select_validation_regime(
    validation_input: PriceValidationInput,
    regime: ValidationRegime,
) -> PriceValidationInput:
    """Select the ordered 900-row core or 100-row stress catalogue regime."""

    if regime == "all":
        return validation_input
    if regime not in {"core", "stress"}:
        raise ValueError(f"Unknown validation regime '{regime}'.")
    expected_count = CORE_ROW_COUNT + STRESS_ROW_COUNT
    if len(validation_input.rows) != expected_count:
        raise ValueError(
            f"Dataset '{validation_input.database_id}' must contain "
            f"{expected_count} ordered rows for a core/stress validation."
        )
    rows = (
        validation_input.rows[:CORE_ROW_COUNT]
        if regime == "core"
        else validation_input.rows[CORE_ROW_COUNT:]
    )
    return PriceValidationInput(
        database_id=validation_input.database_id,
        model_dataset_path=validation_input.model_dataset_path,
        product_dataset_path=validation_input.product_dataset_path,
        rows=rows,
        curve_dataset_path=validation_input.curve_dataset_path,
        monte_carlo_paths_per_price=(
            validation_input.monte_carlo_paths_per_price
        ),
    )


def select_validation_row_ids(
    validation_input: PriceValidationInput,
    row_ids: Sequence[str] | None,
) -> PriceValidationInput:
    """Select explicit fallback rows while preserving catalogue order."""

    if row_ids is None:
        return validation_input
    requested = set(row_ids)
    rows = tuple(row for row in validation_input.rows if row.row_id in requested)
    found = {row.row_id for row in rows}
    missing = requested.difference(found)
    if missing:
        raise ValueError(
            f"Dataset '{validation_input.database_id}' has no rows {sorted(missing)}."
        )
    if not rows:
        raise ValueError("An explicit fallback validation requires at least one row.")
    return PriceValidationInput(
        validation_input.database_id,
        validation_input.model_dataset_path,
        validation_input.product_dataset_path,
        rows,
        validation_input.curve_dataset_path,
        validation_input.monte_carlo_paths_per_price,
    )


def load_parameter_rows(
    dataset_path: str | Path,
    row_field: str,
) -> dict[str, Mapping[str, Any]]:
    """Index one validated parameter dataset by stable row identifier."""

    path = Path(dataset_path)
    document = _read_json(path)
    database_id = _required_string(document, "database_id", str(path))
    rows = document.get(row_field)
    row_count = document.get("row_count")
    if not isinstance(rows, list) or not isinstance(row_count, int):
        raise ValueError(f"Dataset '{database_id}': row_count and {row_field} are required.")
    if row_count <= 0 or len(rows) != row_count:
        raise ValueError(f"Dataset '{database_id}': row_count must match {row_field}.size().")

    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Dataset '{database_id}': every row must be an object.")
        row_id = _required_string(row, "id", f"Dataset '{database_id}'")
        parameters = row.get("parameters")
        if not isinstance(parameters, dict):
            raise ValueError(f"Dataset '{database_id}' row '{row_id}': parameters required.")
        if row_id in indexed:
            raise ValueError(f"Dataset '{database_id}': duplicate row id '{row_id}'.")
        indexed[row_id] = parameters
    return indexed


def summarize_price_comparisons(
    database_id: str,
    comparisons: Sequence[PriceComparison],
    tolerances: ValidationTolerances = ValidationTolerances(),
) -> PriceValidationReport:
    """Aggregate row errors and detect material systematic signed bias."""

    if not comparisons:
        raise ValueError("A QuantLib validation requires at least one price row.")
    errors = [row.generated_price - row.quantlib_price for row in comparisons]
    absolute_errors = [abs(error) for error in errors]
    combined_standard_errors = [
        math.hypot(row.generated_standard_error, row.quantlib_standard_error)
        for row in comparisons
    ]
    relative_errors = [
        absolute_error / max(abs(row.quantlib_price), tolerances.relative_floor)
        for row, absolute_error in zip(comparisons, absolute_errors)
    ]
    row_passed = tuple(
        absolute_error
        <= tolerances.absolute
            + tolerances.relative * abs(row.quantlib_price)
            + tolerances.standard_error_multiplier * standard_error
        for row, absolute_error, standard_error in zip(
            comparisons, absolute_errors, combined_standard_errors
        )
    )
    passed_rows = sum(row_passed)
    allowances = tuple(
        tolerances.absolute
        + tolerances.relative * abs(row.quantlib_price)
        + tolerances.standard_error_multiplier * standard_error
        for row, standard_error in zip(comparisons, combined_standard_errors)
    )
    row_count = len(comparisons)
    mean_error = sum(errors) / row_count
    mean_absolute_error = sum(absolute_errors) / row_count
    rmse = math.sqrt(sum(error * error for error in errors) / row_count)
    standard_error_rmse = math.sqrt(
        sum(value * value for value in combined_standard_errors) / row_count
    )
    if row_count > 1:
        sample_variance = sum(
            (error - mean_error) ** 2 for error in errors
        ) / (row_count - 1)
        standard_error = math.sqrt(sample_variance / row_count)
    else:
        standard_error = 0.0
    mean_reference = sum(abs(row.quantlib_price) for row in comparisons) / row_count
    reported_mean_standard_error = math.sqrt(
        sum(value * value for value in combined_standard_errors)
    ) / row_count
    bias_limit = (
        tolerances.absolute
        + tolerances.relative * mean_reference
        + tolerances.bias_standard_errors
            * max(standard_error, reported_mean_standard_error)
    )
    maximum_absolute_index = max(
        range(row_count), key=absolute_errors.__getitem__
    )
    maximum_relative_index = max(
        range(row_count), key=relative_errors.__getitem__
    )
    return PriceValidationReport(
        database_id=database_id,
        row_count=row_count,
        passed_row_count=passed_rows,
        failed_row_count=row_count - passed_rows,
        failed_row_ids=tuple(
            comparison.row_id
            for comparison, passed in zip(comparisons, row_passed)
            if not passed
        ),
        higher_price_count=sum(error > 0.0 for error in errors),
        lower_price_count=sum(error < 0.0 for error in errors),
        equal_price_count=sum(error == 0.0 for error in errors),
        mean_signed_error=mean_error,
        mean_absolute_error=mean_absolute_error,
        root_mean_squared_error=rmse,
        root_mean_squared_standard_error=standard_error_rmse,
        signed_error_standard_error=standard_error,
        bias_limit=bias_limit,
        systematic_bias=abs(mean_error) > bias_limit,
        maximum_absolute_error=absolute_errors[maximum_absolute_index],
        maximum_absolute_error_row_id=comparisons[maximum_absolute_index].row_id,
        maximum_relative_error=relative_errors[maximum_relative_index],
        maximum_relative_error_row_id=comparisons[maximum_relative_index].row_id,
        row_diagnostics=tuple(
            PriceRowDiagnostic(
                row_id=row.row_id,
                generated_price=row.generated_price,
                reference_price=row.quantlib_price,
                generated_standard_error=row.generated_standard_error,
                reference_standard_error=row.quantlib_standard_error,
                allowance=allowance,
                passed=passed,
            )
            for row, allowance, passed in zip(
                comparisons, allowances, row_passed
            )
        ),
    )


def format_validation_report(report: PriceValidationReport) -> str:
    """Render one compact report suitable for a terminal or CTest log."""

    status = "PASS" if report.passed else "FAIL"
    return "\n".join(
        (
            f"QuantLib validation: {status}",
            f"dataset                  : {report.database_id}",
            f"rows                     : {report.row_count}",
            f"passed / failed          : {report.passed_row_count} / {report.failed_row_count}",
            f"higher / lower / equal   : {report.higher_price_count} / "
            f"{report.lower_price_count} / {report.equal_price_count}",
            f"mean signed error        : {report.mean_signed_error:.12e}",
            f"mean absolute error      : {report.mean_absolute_error:.12e}",
            f"root mean squared error  : {report.root_mean_squared_error:.12e}",
            f"RMS reported std error   : {report.root_mean_squared_standard_error:.12e}",
            f"maximum absolute error   : {report.maximum_absolute_error:.12e} "
            f"(row {report.maximum_absolute_error_row_id})",
            f"maximum relative error   : {report.maximum_relative_error:.12e} "
            f"(row {report.maximum_relative_error_row_id})",
            f"bias limit               : {report.bias_limit:.12e}",
            f"systematic bias          : {'yes' if report.systematic_bias else 'no'}",
        )
    )


ReferencePricer = Callable[
    [Mapping[str, Any], Mapping[str, Any] | None, Mapping[str, Any], PriceResultRow],
    float | tuple[float, float],
]


def run_validation_cli(
    validation: Callable[[str | Path], PriceValidationReport],
    description: str,
) -> int:
    """Run any product validator through the same one-argument CLI."""

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("price_dataset", type=Path)
    arguments = parser.parse_args()
    report = validation(arguments.price_dataset)
    print(format_validation_report(report))
    return 0 if report.passed else 1


def validation_from_reference(
    price_dataset_path: str | Path,
    reference_pricer: ReferencePricer,
    tolerances: ValidationTolerances = ValidationTolerances(),
    require_curve: bool = False,
    regime: ValidationRegime = "all",
    row_ids: Sequence[str] | None = None,
) -> PriceValidationReport:
    """Apply one model/product reference function to every price JSON row."""

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
    curves = None
    if validation_input.curve_dataset_path is not None:
        curves = load_parameter_rows(validation_input.curve_dataset_path, "curves")

    comparisons: list[PriceComparison] = []
    for row in validation_input.rows:
        try:
            model = models[row.model_id]
            product = products[row.product_id]
            curve = curves[row.curve_id] if curves is not None else None
        except KeyError as error:
            raise ValueError(
                f"Price row '{row.row_id}': unknown source row id '{error.args[0]}'."
            ) from error
        try:
            reference = reference_pricer(model, curve, product, row)
        except Exception as error:
            raise RuntimeError(
                f"Price row '{row.row_id}': reference pricing failed: {error}"
            ) from error
        if isinstance(reference, tuple):
            quantlib_price, quantlib_standard_error = reference
        else:
            quantlib_price, quantlib_standard_error = reference, 0.0
        if not math.isfinite(quantlib_price) or not math.isfinite(
            quantlib_standard_error
        ) or quantlib_standard_error < 0.0:
            raise ValueError(
                f"Price row '{row.row_id}': QuantLib returned invalid outputs."
            )
        comparisons.append(
            PriceComparison(
                row_id=row.row_id,
                generated_price=row.generated_price,
                quantlib_price=quantlib_price,
                generated_standard_error=row.generated_standard_error,
                quantlib_standard_error=quantlib_standard_error,
            )
        )
    return summarize_price_comparisons(
        validation_input.database_id, comparisons, tolerances
    )
