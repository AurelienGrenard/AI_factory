"""Black-Scholes path reconstruction and simulation helpers."""

from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.paths.common import load_cpp_library


class CBlackScholesRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("volatility", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
    ]


def _configure_path_library(library: ctypes.CDLL) -> None:
    row_ptr = ctypes.POINTER(CBlackScholesRow)
    double_ptr = ctypes.POINTER(ctypes.c_double)
    seconds_ptr = ctypes.POINTER(ctypes.c_double)
    library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    library.ai_factory_cpu_generate_black_scholes_spot_paths.argtypes = [
        row_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_ptr,
    ]
    library.ai_factory_cpu_generate_black_scholes_spot_paths.restype = ctypes.c_int
    library.ai_factory_generate_black_scholes_spot_paths.argtypes = [
        row_ptr,
        ctypes.c_size_t,
        ctypes.c_size_t,
        double_ptr,
        seconds_ptr,
    ]
    library.ai_factory_generate_black_scholes_spot_paths.restype = ctypes.c_int


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(_resolve(path).read_text(encoding="utf-8"))


def _read_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(_resolve(path).read_text(encoding="utf-8"))


def _parameters_by_id(data: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {row["id"]: row["parameters"] for row in data[key]}


def _cpp_row(
    *,
    seed: int,
    model: dict[str, Any],
    maturity: float,
    strike: float = 1.0,
) -> CBlackScholesRow:
    return CBlackScholesRow(
        float(model["spot"]),
        float(model["risk_free_rate"]),
        float(model.get("dividend_yield", 0.0)),
        float(model["volatility"]),
        float(strike),
        float(maturity),
        int(seed),
    )


def reconstruct_paths_from_context(
    *,
    result_row: dict[str, Any],
    model: dict[str, Any],
    product: dict[str, Any],
    num_paths: int,
    num_steps: int,
    use_gpu: bool,
) -> tuple[list[list[float]], float | None]:
    cpp_row = _cpp_row(
        seed=int(result_row["seed"]),
        model=model,
        maturity=float(product["maturity"]),
        strike=float(product.get("strike", 1.0)),
    )
    output_count = num_paths * (num_steps + 1)
    output_type = ctypes.c_double * output_count
    output = output_type()
    library = load_cpp_library()
    _configure_path_library(library)
    kernel_seconds: float | None = None
    if use_gpu:
        elapsed = ctypes.c_double(0.0)
        status = library.ai_factory_generate_black_scholes_spot_paths(
            ctypes.byref(cpp_row),
            num_paths,
            num_steps,
            output,
            ctypes.byref(elapsed),
        )
        kernel_seconds = float(elapsed.value)
    else:
        status = library.ai_factory_cpu_generate_black_scholes_spot_paths(
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
    use_gpu: bool,
) -> tuple[list[list[float]], float | None]:
    return reconstruct_paths_from_context(
        result_row={"seed": int(seed)},
        model=model,
        product={"maturity": float(maturity), "strike": 1.0},
        num_paths=num_paths,
        num_steps=num_steps,
        use_gpu=use_gpu,
    )


def load_result_context(
    *,
    result_json_path: str | Path,
    result_yaml_path: str | Path,
    row_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    result_data = _read_json(result_json_path)
    result_spec = _read_yaml(result_yaml_path)
    result_row = next(row for row in result_data["results"] if row["id"] == row_id)
    model_data = _read_json(result_spec["model_database"]["json_path"])
    product_data = _read_json(result_spec["product_database"]["json_path"])
    model = _parameters_by_id(model_data, "models")[result_row["model_id"]]
    product = _parameters_by_id(product_data, "products")[result_row["product_id"]]
    return result_row, model, product, result_spec


def reconstruct_paths_for_result(
    *,
    result_json_path: str | Path,
    result_yaml_path: str | Path,
    row_id: str,
    num_paths: int,
    num_steps: int,
    use_gpu: bool,
) -> tuple[list[list[float]], float | None]:
    result_row, model, product, _ = load_result_context(
        result_json_path=result_json_path,
        result_yaml_path=result_yaml_path,
        row_id=row_id,
    )
    return reconstruct_paths_from_context(
        result_row=result_row,
        model=model,
        product=product,
        num_paths=num_paths,
        num_steps=num_steps,
        use_gpu=use_gpu,
    )
