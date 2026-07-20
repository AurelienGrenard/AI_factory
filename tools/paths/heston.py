"""Heston path reconstruction, simulation, and repricing helpers."""

from __future__ import annotations

import ctypes
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paths.common import load_cpp_library, step_count_for_maturity  # noqa: E402

HESTON_SCHEME = "qe_martingale"
HESTON_SCHEME_CODES = {
    "euler_full_truncation": 0,
    "qe_martingale": 2,
}


class CHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
        ("scheme", ctypes.c_int),
    ]
 


def _configure_path_library(library: ctypes.CDLL) -> None:
    row_ptr = ctypes.POINTER(CHestonRow)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    seconds_ptr = ctypes.POINTER(ctypes.c_double)
    library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    try:
        library.ai_factory_cpu_generate_heston_spot_paths.argtypes = [
            row_ptr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            double_ptr,
        ]
        library.ai_factory_cpu_generate_heston_spot_paths.restype = ctypes.c_int
        library.ai_factory_generate_heston_spot_paths.argtypes = [
            row_ptr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            double_ptr,
            seconds_ptr,
        ]
        library.ai_factory_generate_heston_spot_paths.restype = ctypes.c_int
        library.ai_factory_cpu_generate_heston_state_paths.argtypes = [
            row_ptr, ctypes.c_size_t, ctypes.c_size_t, double_ptr, double_ptr,
        ]
        library.ai_factory_cpu_generate_heston_state_paths.restype = ctypes.c_int
        library.ai_factory_generate_heston_state_paths.argtypes = [
            row_ptr, ctypes.c_size_t, ctypes.c_size_t,
            double_ptr, double_ptr, seconds_ptr,
        ]
        library.ai_factory_generate_heston_state_paths.restype = ctypes.c_int
    except AttributeError as error:
        raise RuntimeError(
            "Loaded libai_factory_cuda.so does not expose the Heston spot-path "
            "export API. Rebuild the C++ shared library."
        ) from error


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _parameters_by_id(data: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {row["id"]: row["parameters"] for row in data[key]}


def _cpp_scheme_code(scheme: str) -> int:
    try:
        return HESTON_SCHEME_CODES[scheme]
    except KeyError as error:
        raise ValueError(f"Unsupported Heston scheme: {scheme}") from error


def _cpp_row(
    row: dict[str, Any],
    model: dict[str, Any],
    product: dict[str, Any],
    scheme: str,
) -> CHestonRow:
    return CHestonRow(
        float(model["spot"]),
        float(model["risk_free_rate"]),
        float(model.get("dividend_yield", 0.0)),
        float(model["initial_variance"]),
        float(model["kappa"]),
        float(model["theta"]),
        float(model["volatility_of_variance"]),
        float(model["rho"]),
        float(product.get("strike", product.get("volatility_strike", 1.0))),
        float(product["maturity"]),
        int(row["seed"]),
        _cpp_scheme_code(scheme),
    )


def load_result_context(
    *,
    result_json_path: str | Path,
    result_yaml_path: str | Path,
    row_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a stored result row and its model/product parameter rows."""

    result_data = _read_json(_resolve(result_json_path))
    result_spec = _read_yaml(_resolve(result_yaml_path))
    result_row = next(row for row in result_data["results"] if row["id"] == row_id)

    model_json = _resolve(result_spec["model_database"]["json_path"])
    product_json = _resolve(result_spec["product_database"]["json_path"])
    model_data = _read_json(model_json)
    product_data = _read_json(product_json)
    model = _parameters_by_id(model_data, "models")[result_row["model_id"]]
    product = _parameters_by_id(product_data, "products")[result_row["product_id"]]
    return result_row, model, product, result_spec


def reconstruct_paths_from_context(
    *,
    result_row: dict[str, Any],
    model: dict[str, Any],
    product: dict[str, Any],
    num_paths: int,
    num_steps: int,
    use_gpu: bool,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[list[float]], float | None]:
    """Reconstruct the exact C++ Philox spot paths for one Heston result row."""

    cpp_product = dict(product)
    if "strike" not in cpp_product and "volatility_strike" in cpp_product:
        cpp_product["strike"] = cpp_product["volatility_strike"]
    cpp_row = _cpp_row(result_row, model, cpp_product, heston_scheme)
    output_count = num_paths * (num_steps + 1)
    output_type = ctypes.c_double * output_count
    output = output_type()
    library = load_cpp_library()
    _configure_path_library(library)

    kernel_seconds: float | None = None
    if use_gpu:
        elapsed = ctypes.c_double(0.0)
        status = library.ai_factory_generate_heston_spot_paths(
            ctypes.byref(cpp_row),
            num_paths,
            num_steps,
            output,
            ctypes.byref(elapsed),
        )
        kernel_seconds = float(elapsed.value)
    else:
        status = library.ai_factory_cpu_generate_heston_spot_paths(
            ctypes.byref(cpp_row),
            num_paths,
            num_steps,
            output,
        )
    if status != 0:
        error = library.ai_factory_cuda_last_error()
        raise RuntimeError(error.decode() if error else "Unknown C++ path error")

    step_count = num_steps + 1
    return [
        [float(output[path * step_count + step]) for step in range(step_count)]
        for path in range(num_paths)
    ], kernel_seconds


def reconstruct_state_paths_from_context(
    *,
    result_row: dict[str, Any],
    model: dict[str, Any],
    product: dict[str, Any],
    num_paths: int,
    num_steps: int,
    use_gpu: bool,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Reconstruct exact spot and variance paths used by Heston LSM."""

    cpp_row = _cpp_row(result_row, model, product, heston_scheme)
    output_count = num_paths * (num_steps + 1)
    output_type = ctypes.c_double * output_count
    spots = output_type()
    variances = output_type()
    library = load_cpp_library()
    _configure_path_library(library)
    kernel_seconds: float | None = None
    if use_gpu:
        elapsed = ctypes.c_double(0.0)
        status = library.ai_factory_generate_heston_state_paths(
            ctypes.byref(cpp_row), num_paths, num_steps,
            spots, variances, ctypes.byref(elapsed),
        )
        kernel_seconds = float(elapsed.value)
    else:
        status = library.ai_factory_cpu_generate_heston_state_paths(
            ctypes.byref(cpp_row), num_paths, num_steps, spots, variances
        )
    if status != 0:
        error = library.ai_factory_cuda_last_error()
        raise RuntimeError(error.decode() if error else "Unknown C++ state-path error")
    shape = (num_paths, num_steps + 1)
    spot_paths = np.ctypeslib.as_array(spots).copy().reshape(shape)
    variance_paths = np.ctypeslib.as_array(variances).copy().reshape(shape)
    return spot_paths, variance_paths, kernel_seconds


def simulate_paths(
    *,
    model: dict[str, Any],
    maturity: float,
    seed: int,
    num_paths: int,
    num_steps: int,
    use_gpu: bool,
    heston_scheme: str = HESTON_SCHEME,
) -> tuple[list[list[float]], float | None]:
    """Generate Heston spot paths directly from model parameters.

    This is the model-only entry point: no product database and no result row are
    required. The dummy strike is ignored by the path-export kernel.
    """

    result_row = {"seed": int(seed)}
    product = {"strike": 1.0, "maturity": float(maturity)}
    return reconstruct_paths_from_context(
        result_row=result_row,
        model=model,
        product=product,
        num_paths=num_paths,
        num_steps=num_steps,
        use_gpu=use_gpu,
        heston_scheme=heston_scheme,
    )


def reconstruct_paths_for_result(
    *,
    result_json_path: str | Path,
    result_yaml_path: str | Path,
    row_id: str,
    use_gpu: bool,
    num_paths: int | None = None,
    num_steps: int | None = None,
) -> tuple[list[list[float]], float | None, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load a result row, reconstruct its paths, and return the full context."""

    result_row, model, product, spec = load_result_context(
        result_json_path=result_json_path,
        result_yaml_path=result_yaml_path,
        row_id=row_id,
    )
    if num_paths is None:
        num_paths = int(spec["summary"]["num_paths"])
    if num_steps is None:
        num_steps = step_count_for_maturity(
            float(product["maturity"]),
            spec["time_grid"]["target_dt"],
        )
    paths, kernel_seconds = reconstruct_paths_from_context(
        result_row=result_row,
        model=model,
        product=product,
        num_paths=num_paths,
        num_steps=num_steps,
        use_gpu=use_gpu,
        heston_scheme=str(spec["summary"].get("heston_scheme", HESTON_SCHEME)),
    )
    return paths, kernel_seconds, result_row, model, product, spec


def reconstruct_state_paths_for_result(
    *,
    result_json_path: str | Path,
    result_yaml_path: str | Path,
    row_id: str,
    use_gpu: bool,
) -> tuple[
    np.ndarray, np.ndarray, float | None,
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any],
]:
    result_row, model, product, spec = load_result_context(
        result_json_path=result_json_path,
        result_yaml_path=result_yaml_path,
        row_id=row_id,
    )
    num_paths = int(spec["summary"]["num_paths"])
    num_steps = step_count_for_maturity(
        float(product["maturity"]), spec["time_grid"]["target_dt"]
    )
    spots, variances, seconds = reconstruct_state_paths_from_context(
        result_row=result_row,
        model=model,
        product=product,
        num_paths=num_paths,
        num_steps=num_steps,
        use_gpu=use_gpu,
        heston_scheme=str(spec["summary"].get("heston_scheme", HESTON_SCHEME)),
    )
    return spots, variances, seconds, result_row, model, product, spec


def price_lookback_from_paths(
    paths: list[list[float]],
    *,
    strike: float,
    maturity: float,
    rate: float,
) -> dict[str, float]:
    """Price a fixed-strike lookback call from reconstructed spot paths."""

    discounted = [
        math.exp(-rate * maturity) * max(max(path) - strike, 0.0)
        for path in paths
    ]
    price = sum(discounted) / len(discounted)
    if len(discounted) < 2:
        return {"price": price, "standard_error": 0.0}
    sumsq = sum(value * value for value in discounted)
    variance = (sumsq - len(discounted) * price * price) / (len(discounted) - 1)
    return {
        "price": price,
        "standard_error": math.sqrt(max(variance, 0.0)) / math.sqrt(len(discounted)),
    }


def price_asian_arithmetic_from_paths(
    paths: list[list[float]],
    *,
    strike: float,
    maturity: float,
    rate: float,
) -> dict[str, float]:
    """Price an arithmetic Asian call from reconstructed spot paths."""

    discounted = []
    for path in paths:
        if len(path) < 2:
            raise ValueError("Asian path repricing requires at least one monitoring date.")
        average = sum(path[1:]) / float(len(path) - 1)
        discounted.append(math.exp(-rate * maturity) * max(average - strike, 0.0))
    price = sum(discounted) / len(discounted)
    if len(discounted) < 2:
        return {"price": price, "standard_error": 0.0}
    sumsq = sum(value * value for value in discounted)
    variance = (sumsq - len(discounted) * price * price) / (len(discounted) - 1)
    return {
        "price": price,
        "standard_error": math.sqrt(max(variance, 0.0)) / math.sqrt(len(discounted)),
    }


def price_volatility_swap_from_paths(
    paths: list[list[float]],
    *,
    volatility_strike: float,
    maturity: float,
    rate: float,
    observations_per_year: float = 52.0,
) -> dict[str, float]:
    """Price a weekly realized-volatility swap from reconstructed spot paths."""

    discounted = []
    for path in paths:
        if len(path) < 2:
            raise ValueError("Volatility swap repricing requires at least one return.")
        sum_squared_log_returns = 0.0
        for previous, current in zip(path, path[1:]):
            log_return = math.log(current / previous)
            sum_squared_log_returns += log_return * log_return
        realized_volatility = math.sqrt(
            observations_per_year / float(len(path) - 1)
            * sum_squared_log_returns
        )
        discounted.append(
            math.exp(-rate * maturity)
            * (realized_volatility - volatility_strike)
        )
    price = sum(discounted) / len(discounted)
    if len(discounted) < 2:
        return {"price": price, "standard_error": 0.0}
    sumsq = sum(value * value for value in discounted)
    variance = (sumsq - len(discounted) * price * price) / (len(discounted) - 1)
    return {
        "price": price,
        "standard_error": math.sqrt(max(variance, 0.0)) / math.sqrt(len(discounted)),
    }


def _heston_state_basis(
    spot: float, variance: float, strike: float, theta: float
) -> tuple[float, float, float, float, float, float]:
    value = spot / strike
    scaled_variance = variance / theta
    l1 = 1.0 - value
    return (
        1.0,
        l1,
        1.0 - 2.0 * value + 0.5 * value * value,
        scaled_variance,
        scaled_variance * scaled_variance,
        l1 * scaled_variance,
    )


def price_american_put_from_paths(
    paths: list[list[float]],
    *,
    variance_paths: list[list[float]],
    theta: float,
    strike: float,
    maturity: float,
    rate: float,
) -> dict[str, float]:
    """Longstaff-Schwartz price from reconstructed Heston state paths."""

    path_array = np.asarray(paths, dtype=float)
    variance_array = np.asarray(variance_paths, dtype=float)
    if path_array.ndim != 2 or path_array.shape[1] < 2:
        raise ValueError("American put repricing requires at least two dates.")
    path_count, date_count = path_array.shape
    step_count = date_count - 1
    dt = maturity / float(step_count)
    payoffs = np.maximum(strike - path_array, 0.0)
    cashflows = payoffs[:, -1].copy()
    exercise_steps = np.full(path_count, step_count, dtype=np.int64)
    if variance_array.shape != path_array.shape:
        raise ValueError("Heston variance paths must match spot paths.")
    basis_count = 6

    for step in range(step_count - 1, 0, -1):
        immediate = payoffs[:, step]
        itm = immediate > 0.0
        if int(itm.sum()) <= basis_count:
            continue
        value = path_array[itm, step] / strike
        scaled_variance = variance_array[itm, step] / theta
        l1 = 1.0 - value
        design = np.column_stack((
            np.ones_like(value),
            l1,
            1.0 - 2.0 * value + 0.5 * value * value,
            scaled_variance,
            scaled_variance * scaled_variance,
            l1 * scaled_variance,
        ))
        target = cashflows[itm] * np.exp(-rate * dt * (exercise_steps[itm] - step))
        normal = design.T @ design
        ridge = 1.0e-10 * np.trace(normal) / float(basis_count)
        coefficients = np.linalg.solve(
            normal + ridge * np.eye(basis_count), design.T @ target
        )
        continuation = design @ coefficients
        exercise = np.zeros(path_count, dtype=bool)
        exercise_indices = np.flatnonzero(itm)
        exercise[exercise_indices] = immediate[itm] > continuation
        cashflows[exercise] = immediate[exercise]
        exercise_steps[exercise] = step

    discounted = cashflows * np.exp(-rate * dt * exercise_steps)
    price = float(discounted.mean())
    if path_count < 2:
        return {"price": price, "standard_error": 0.0}
    standard_error = float(discounted.std(ddof=1) / math.sqrt(path_count))
    return {"price": price, "standard_error": standard_error}
