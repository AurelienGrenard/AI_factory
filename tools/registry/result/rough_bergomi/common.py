"""Shared native Rough Bergomi result execution helpers."""

from __future__ import annotations

import ctypes
from time import perf_counter
from typing import Any

from tools.common.native_library import load_cpp_library
from tools.common.time_grid import step_count_for_maturity


def _last_error(library: ctypes.CDLL) -> str:
    function = library.ai_factory_cuda_last_error
    function.argtypes = []
    function.restype = ctypes.c_char_p
    raw = function()
    return raw.decode("utf-8") if raw else "unknown C++ error"


def _rough_row(
    row_type: type[ctypes.Structure],
    row: dict[str, Any],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
) -> ctypes.Structure:
    model = model_by_id[row["model_id"]]
    product = product_by_id[row["product_id"]]
    strike = product.get("strike", product.get("volatility_strike"))
    if strike is None:
        raise ValueError("Rough Bergomi product row requires a strike.")
    return row_type(
        float(model["spot"]),
        float(model["risk_free_rate"]),
        float(model.get("dividend_yield", 0.0)),
        float(model["forward_variance"]),
        float(model["eta"]),
        float(model["hurst"]) - 0.5,
        float(model["correlation"]),
        float(strike),
        float(product["maturity"]),
        int(row["seed"]),
    )


def cpp_outputs_for_time_grid(
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    row_type: type[ctypes.Structure],
    output_type: type[ctypes.Structure],
    product_symbol: str,
    num_paths: int,
    target_dt: str | float | int,
    use_gpu: bool,
    relative_bump: float | None = None,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    delta_crn = relative_bump is not None
    delta_suffix = "_delta_crn" if delta_crn else ""
    device_suffix = "gpu" if use_gpu else "cpu"
    symbol = (
        f"ai_factory_price_rough_bergomi_{product_symbol}"
        f"{delta_suffix}_{device_suffix}_batch"
    )
    library = load_cpp_library()
    function = getattr(library, symbol)
    args: list[Any] = [
        ctypes.POINTER(row_type),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    if delta_crn:
        args.append(ctypes.c_double)
    args.append(ctypes.POINTER(output_type))
    if use_gpu:
        args.append(ctypes.POINTER(ctypes.c_double))
    function.argtypes = args
    function.restype = ctypes.c_int

    groups: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for index, row in enumerate(rows):
        maturity = float(product_by_id[row["product_id"]]["maturity"])
        groups.setdefault(step_count_for_maturity(maturity, target_dt), []).append((index, row))

    outputs: list[dict[str, float] | None] = [None] * len(rows)
    kernel_seconds = 0.0
    started = perf_counter()
    for num_steps, indexed_rows in groups.items():
        group_rows = [row for _, row in indexed_rows]
        row_array_type = row_type * len(group_rows)
        output_array_type = output_type * len(group_rows)
        row_array = row_array_type(
            *[
                _rough_row(row_type, row, model_by_id, product_by_id)
                for row in group_rows
            ]
        )
        output_array = output_array_type()
        arguments: list[Any] = [
            row_array,
            len(group_rows),
            num_paths,
            num_steps,
        ]
        if delta_crn:
            arguments.append(relative_bump)
        arguments.append(output_array)
        elapsed = ctypes.c_double(0.0)
        if use_gpu:
            arguments.append(ctypes.byref(elapsed))
        status = function(*arguments)
        if status != 0:
            raise RuntimeError(_last_error(library))
        kernel_seconds += float(elapsed.value)
        for (index, _), output in zip(indexed_rows, output_array, strict=True):
            result = {
                "price": float(output.price),
                "standard_error": float(output.standard_error),
            }
            if delta_crn:
                result.update(
                    {
                        "delta": float(output.delta),
                        "delta_standard_error": float(output.delta_standard_error),
                    }
                )
            outputs[index] = result

    timing = {"wall_seconds": perf_counter() - started}
    if use_gpu:
        timing["kernel_seconds"] = kernel_seconds
    return [output for output in outputs if output is not None], timing
