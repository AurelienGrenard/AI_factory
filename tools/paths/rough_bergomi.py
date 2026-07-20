"""Rough Bergomi path reconstruction, simulation, and repricing helpers."""

from __future__ import annotations

import ctypes
import json
import math
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paths.common import load_cpp_library, step_count_for_maturity  # noqa: E402


class CRoughBergomiRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("forward_variance", ctypes.c_double),
        ("eta", ctypes.c_double),
        ("alpha", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
    ]


def _configure_path_library(library: ctypes.CDLL) -> None:
    row_ptr = ctypes.POINTER(CRoughBergomiRow)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    seconds_ptr = ctypes.POINTER(ctypes.c_double)
    library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    try:
        library.ai_factory_cpu_generate_rough_bergomi_spot_paths.argtypes = [
            row_ptr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            double_ptr,
        ]
        library.ai_factory_cpu_generate_rough_bergomi_spot_paths.restype = ctypes.c_int
        library.ai_factory_generate_rough_bergomi_spot_paths.argtypes = [
            row_ptr,
            ctypes.c_size_t,
            ctypes.c_size_t,
            double_ptr,
            seconds_ptr,
        ]
        library.ai_factory_generate_rough_bergomi_spot_paths.restype = ctypes.c_int
    except AttributeError as error:
        raise RuntimeError(
            "Loaded libai_factory_cuda.so does not expose the Rough Bergomi "
            "spot-path export API. Rebuild the C++ shared library."
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


def _rough_alpha(model: dict[str, Any]) -> float:
    if "alpha" in model:
        return float(model["alpha"])
    return float(model["hurst"]) - 0.5


def _rough_correlation(model: dict[str, Any]) -> float:
    if "rho" in model:
        return float(model["rho"])
    return float(model["correlation"])


def _cpp_row(
    row: dict[str, Any],
    model: dict[str, Any],
    product: dict[str, Any],
) -> CRoughBergomiRow:
    return CRoughBergomiRow(
        float(model["spot"]),
        float(model["risk_free_rate"]),
        float(model.get("dividend_yield", 0.0)),
        float(model["forward_variance"]),
        float(model["eta"]),
        _rough_alpha(model),
        _rough_correlation(model),
        float(product.get("strike", product.get("volatility_strike", 1.0))),
        float(product["maturity"]),
        int(row["seed"]),
    )


def load_result_context(
    *,
    result_json_path: str | Path,
    result_yaml_path: str | Path,
    row_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    use_gpu: bool = False,
) -> tuple[list[list[float]], float | None]:
    """Reconstruct exact C++ Philox spot paths for one Rough Bergomi result row."""

    cpp_product = dict(product)
    if "strike" not in cpp_product and "volatility_strike" in cpp_product:
        cpp_product["strike"] = cpp_product["volatility_strike"]
    cpp_row = _cpp_row(result_row, model, cpp_product)
    output_count = num_paths * (num_steps + 1)
    output_type = ctypes.c_double * output_count
    output = output_type()
    library = load_cpp_library()
    _configure_path_library(library)

    kernel_seconds: float | None = None
    if use_gpu:
        elapsed = ctypes.c_double(0.0)
        status = library.ai_factory_generate_rough_bergomi_spot_paths(
            ctypes.byref(cpp_row),
            num_paths,
            num_steps,
            output,
            ctypes.byref(elapsed),
        )
        kernel_seconds = float(elapsed.value)
    else:
        status = library.ai_factory_cpu_generate_rough_bergomi_spot_paths(
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


def simulate_paths(
    *,
    model: dict[str, Any],
    maturity: float,
    seed: int,
    num_paths: int,
    num_steps: int,
    use_gpu: bool = False,
) -> tuple[list[list[float]], float | None]:
    """Generate Rough Bergomi spot paths directly from model parameters.

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
    )


def reconstruct_paths_for_result(
    *,
    result_json_path: str | Path,
    result_yaml_path: str | Path,
    row_id: str,
    use_gpu: bool = False,
    num_paths: int | None = None,
    num_steps: int | None = None,
) -> tuple[list[list[float]], float | None, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    )
    return paths, kernel_seconds, result_row, model, product, spec


def price_lookback_from_paths(
    paths: list[list[float]],
    *,
    strike: float,
    maturity: float,
    rate: float,
) -> dict[str, float]:
    discounted = [
        math.exp(-rate * maturity) * max(max(path) - strike, 0.0)
        for path in paths
    ]
    return _summary(discounted)


def price_asian_arithmetic_from_paths(
    paths: list[list[float]],
    *,
    strike: float,
    maturity: float,
    rate: float,
) -> dict[str, float]:
    discounted = []
    for path in paths:
        if len(path) < 2:
            raise ValueError("Asian path repricing requires at least one monitoring date.")
        average = sum(path[1:]) / float(len(path) - 1)
        discounted.append(math.exp(-rate * maturity) * max(average - strike, 0.0))
    return _summary(discounted)


def price_volatility_swap_from_paths(
    paths: list[list[float]],
    *,
    volatility_strike: float,
    maturity: float,
    rate: float,
    observations_per_year: float = 52.0,
) -> dict[str, float]:
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
            math.exp(-rate * maturity) * (realized_volatility - volatility_strike)
        )
    return _summary(discounted)


def _summary(discounted: list[float]) -> dict[str, float]:
    price = sum(discounted) / len(discounted)
    if len(discounted) < 2:
        return {"price": price, "standard_error": 0.0}
    sumsq = sum(value * value for value in discounted)
    variance = (sumsq - len(discounted) * price * price) / (len(discounted) - 1)
    return {
        "price": price,
        "standard_error": math.sqrt(max(variance, 0.0)) / math.sqrt(len(discounted)),
    }
