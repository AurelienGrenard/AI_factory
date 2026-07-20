"""Shared orchestration for model-specific autocall source engines."""

from __future__ import annotations

import ctypes
import math
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SRC_PYTHON = PROJECT_ROOT / "src_python"
if str(SRC_PYTHON) not in sys.path:
    sys.path.insert(0, str(SRC_PYTHON))

from ai_factory.pytorch.black_scholes import autocalls as black_scholes_pytorch
from ai_factory.pytorch.heston import autocalls as heston_pytorch
from ai_factory.pytorch.rough_bergomi import autocalls as rough_bergomi_pytorch
from ai_factory.pytorch.rough_heston import autocalls as rough_heston_pytorch
from tools.common.native_library import load_cpp_library
from tools.common.time_grid import step_count_for_maturity

DEFAULT_NUM_PATHS = 16_384
DEFAULT_TARGET_DT = "1/52"
DEFAULT_FIRST_SEED = 934_100_001


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


class CRoughHestonRow(ctypes.Structure):
    _fields_ = [
        ("spot", ctypes.c_double),
        ("risk_free_rate", ctypes.c_double),
        ("dividend_yield", ctypes.c_double),
        ("initial_variance", ctypes.c_double),
        ("kappa", ctypes.c_double),
        ("theta", ctypes.c_double),
        ("volatility_of_variance", ctypes.c_double),
        ("hurst", ctypes.c_double),
        ("rho", ctypes.c_double),
        ("strike", ctypes.c_double),
        ("maturity", ctypes.c_double),
        ("seed", ctypes.c_uint64),
    ]


class CAutocallTerms(ctypes.Structure):
    _fields_ = [
        ("autocall_barrier", ctypes.c_double),
        ("coupon_barrier", ctypes.c_double),
        ("protection_barrier", ctypes.c_double),
        ("coupon_rate", ctypes.c_double),
        ("observation_count", ctypes.c_size_t),
        ("first_autocall_observation", ctypes.c_size_t),
    ]


class CBlackScholesAutocallRow(ctypes.Structure):
    _fields_ = [("model", CBlackScholesRow), ("product", CAutocallTerms)]


class CHestonAutocallRow(ctypes.Structure):
    _fields_ = [("model", CHestonRow), ("product", CAutocallTerms)]


class CRoughBergomiAutocallRow(ctypes.Structure):
    _fields_ = [("model", CRoughBergomiRow), ("product", CAutocallTerms)]


class CRoughHestonAutocallRow(ctypes.Structure):
    _fields_ = [("model", CRoughHestonRow), ("product", CAutocallTerms)]


class CAutocallOutput(ctypes.Structure):
    _fields_ = [
        ("price", ctypes.c_double),
        ("standard_error", ctypes.c_double),
        ("autocall_probability", ctypes.c_double),
        ("mean_autocall_time", ctypes.c_double),
        ("maturity_probability", ctypes.c_double),
        ("coupon_payment_frequency", ctypes.c_double),
        ("mean_total_coupon", ctypes.c_double),
        ("capital_loss_probability", ctypes.c_double),
        ("mean_redemption_given_loss", ctypes.c_double),
    ]


MODEL_CONFIG = {
    "black_scholes": (CBlackScholesAutocallRow, black_scholes_pytorch),
    "heston": (CHestonAutocallRow, heston_pytorch),
    "rough_bergomi": (CRoughBergomiAutocallRow, rough_bergomi_pytorch),
    "rough_heston": (CRoughHestonAutocallRow, rough_heston_pytorch),
}


def _terms(product: dict[str, Any]) -> CAutocallTerms:
    return CAutocallTerms(
        float(product["autocall_barrier"]),
        float(product["coupon_barrier"]),
        float(product["protection_barrier"]),
        float(product["coupon_rate_per_observation"]),
        int(product["observation_count"]),
        int(product["first_autocall_observation"]),
    )


def _native_row(
    model_family: str,
    row: dict[str, Any],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
):
    model = model_by_id[row["model_id"]]
    product = product_by_id[row["product_id"]]
    maturity = float(product["maturity"])
    seed = int(row["seed"])
    if model_family == "black_scholes":
        native_model = CBlackScholesRow(
            float(model["spot"]),
            float(model["risk_free_rate"]),
            float(model["dividend_yield"]),
            float(model["volatility"]),
            0.0,
            maturity,
            seed,
        )
        return CBlackScholesAutocallRow(native_model, _terms(product))
    if model_family == "heston":
        native_model = CHestonRow(
            float(model["spot"]),
            float(model["risk_free_rate"]),
            float(model["dividend_yield"]),
            float(model["initial_variance"]),
            float(model["kappa"]),
            float(model["theta"]),
            float(model["volatility_of_variance"]),
            float(model["rho"]),
            0.0,
            maturity,
            seed,
            2,
        )
        return CHestonAutocallRow(native_model, _terms(product))
    if model_family == "rough_heston":
        native_model = CRoughHestonRow(
            float(model["spot"]),
            float(model["risk_free_rate"]),
            float(model["dividend_yield"]),
            float(model["initial_variance"]),
            float(model["kappa"]),
            float(model["theta"]),
            float(model["volatility_of_variance"]),
            float(model["hurst"]),
            float(model["rho"]),
            0.0,
            maturity,
            seed,
        )
        return CRoughHestonAutocallRow(native_model, _terms(product))
    native_model = CRoughBergomiRow(
        float(model["spot"]),
        float(model["risk_free_rate"]),
        float(model["dividend_yield"]),
        float(model["forward_variance"]),
        float(model["eta"]),
        float(model["hurst"]) - 0.5,
        float(model["rho"] if "rho" in model else model["correlation"]),
        0.0,
        maturity,
        seed,
    )
    return CRoughBergomiAutocallRow(native_model, _terms(product))


def _diagnostic(native: CAutocallOutput) -> dict[str, float]:
    return {
        name: float(getattr(native, name))
        for name, _ in CAutocallOutput._fields_[2:]
    }


def _groups(
    rows: list[dict[str, Any]],
    product_by_id: dict[str, Any],
    target_dt: str | float,
) -> list[tuple[int, list[tuple[int, dict[str, Any]]]]]:
    grouped: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        maturity = float(product_by_id[row["product_id"]]["maturity"])
        grouped[step_count_for_maturity(maturity, target_dt)].append((index, row))
    return sorted(grouped.items())


def cpp_outputs(
    model_family: str,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float,
    use_gpu: bool,
    include_diagnostics: bool = False,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    row_type, _ = MODEL_CONFIG[model_family]
    library = load_cpp_library()
    function = getattr(
        library,
        f"ai_factory_price_{model_family}_autocall_{'gpu' if use_gpu else 'cpu'}_batch",
    )
    args = [
        ctypes.POINTER(row_type),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(CAutocallOutput),
    ]
    if use_gpu:
        args.append(ctypes.POINTER(ctypes.c_double))
    function.argtypes = args
    function.restype = ctypes.c_int
    library.ai_factory_cuda_last_error.restype = ctypes.c_char_p
    ordered: list[dict[str, float] | None] = [None] * len(rows)
    kernel_seconds = 0.0
    started = perf_counter()
    for num_steps, indexed_rows in _groups(rows, product_by_id, target_dt):
        group_rows = [row for _, row in indexed_rows]
        native_rows = (row_type * len(group_rows))(
            *[
                _native_row(model_family, row, model_by_id, product_by_id)
                for row in group_rows
            ]
        )
        native_outputs = (CAutocallOutput * len(group_rows))()
        if use_gpu:
            group_kernel = ctypes.c_double(0.0)
            status = function(
                native_rows,
                len(group_rows),
                num_paths,
                num_steps,
                native_outputs,
                ctypes.byref(group_kernel),
            )
            kernel_seconds += float(group_kernel.value)
        else:
            status = function(
                native_rows, len(group_rows), num_paths, num_steps, native_outputs
            )
        if status != 0:
            raw = library.ai_factory_cuda_last_error()
            raise RuntimeError(raw.decode("utf-8") if raw else "Native autocall error")
        for (index, _), native in zip(indexed_rows, native_outputs, strict=True):
            output = {
                "price": float(native.price),
                "standard_error": float(native.standard_error),
            }
            if include_diagnostics:
                output.update(_diagnostic(native))
            ordered[index] = output
    timing = {"wall_seconds": perf_counter() - started}
    if use_gpu:
        timing["kernel_seconds"] = kernel_seconds
    return [output for output in ordered if output is not None], timing


def python_outputs(
    model_family: str,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int,
    target_dt: str | float,
    device: str,
    **_: Any,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    _, module = MODEL_CONFIG[model_family]
    ordered: list[dict[str, float] | None] = [None] * len(rows)
    started = perf_counter()
    for num_steps, indexed_rows in _groups(rows, product_by_id, target_dt):
        group_rows = [row for _, row in indexed_rows]
        group_outputs, _ = module.price_batch(
            group_rows,
            model_by_id,
            product_by_id,
            num_paths=num_paths,
            num_steps=num_steps,
            device=device,
        )
        for (index, _), output in zip(indexed_rows, group_outputs, strict=True):
            ordered[index] = output
    return [output for output in ordered if output is not None], {
        "wall_seconds": perf_counter() - started
    }


def economic_diagnostics(
    model_family: str,
    rows: list[dict[str, Any]],
    model_by_id: dict[str, Any],
    product_by_id: dict[str, Any],
    *,
    num_paths: int = DEFAULT_NUM_PATHS,
    target_dt: str | float = DEFAULT_TARGET_DT,
) -> list[dict[str, float]]:
    outputs, _ = cpp_outputs(
        model_family,
        rows,
        model_by_id,
        product_by_id,
        num_paths=num_paths,
        target_dt=target_dt,
        use_gpu=True,
        include_diagnostics=True,
    )
    return [
        {key: value for key, value in output.items() if key not in {"price", "standard_error"}}
        for output in outputs
    ]


def price_from_paths(
    paths: list[list[float]],
    *,
    model: dict[str, Any],
    product: dict[str, Any],
) -> dict[str, float]:
    """Reprice one autocall row from reconstructed weekly spot paths."""

    observation_count = int(product["observation_count"])
    num_steps = len(paths[0]) - 1
    if num_steps % observation_count != 0:
        raise ValueError("Observation count must divide the reconstructed step count.")
    stride = num_steps // observation_count
    spot0 = float(model["spot"])
    rate = float(model["risk_free_rate"])
    maturity = float(product["maturity"])
    discounted_payoffs: list[float] = []
    for path in paths:
        unpaid = 0
        payoff = 0.0
        called = False
        for observation in range(1, observation_count + 1):
            performance = float(path[observation * stride]) / spot0
            time = maturity * observation / observation_count
            unpaid += 1
            cash = 0.0
            if performance >= float(product["coupon_barrier"]):
                cash = float(product["coupon_rate_per_observation"]) * unpaid
                unpaid = 0
            if (
                observation >= int(product["first_autocall_observation"])
                and performance >= float(product["autocall_barrier"])
            ):
                cash += 1.0
                called = True
            payoff += math.exp(-rate * time) * cash
            if called:
                break
        if not called:
            terminal_performance = float(path[-1]) / spot0
            redemption = (
                terminal_performance
                if terminal_performance < float(product["protection_barrier"])
                else 1.0
            )
            payoff += math.exp(-rate * maturity) * redemption
        discounted_payoffs.append(payoff)
    count = len(discounted_payoffs)
    mean = sum(discounted_payoffs) / count
    variance = sum((value - mean) ** 2 for value in discounted_payoffs) / (count - 1)
    return {
        "price": mean,
        "standard_error": math.sqrt(max(variance, 0.0) / count),
    }
