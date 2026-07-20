"""Native row mapping and scheme conventions shared by Heston result recipes."""

from __future__ import annotations

import ctypes
from typing import Any

from tools.common.native_library import load_cpp_library as _load_cpp_library

HESTON_SCHEME = "qe_martingale"
HESTON_SCHEME_CODES = {
    "euler": 0,
    "euler_full_truncation": 0,
    "qe": 1,
    "andersen_qe": 1,
    "qe_martingale": 2,
    "andersen_qe_martingale": 2,
    "qe-m": 2,
}


def load_cpp_library() -> ctypes.CDLL:
    return _load_cpp_library()


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


def cpp_scheme_code(scheme: str) -> int:
    try:
        return HESTON_SCHEME_CODES[scheme]
    except KeyError as error:
        raise ValueError(f"Unsupported Heston scheme: {scheme}") from error


def cpp_row(
    row: dict[str, Any],
    model: dict[str, Any],
    product: dict[str, Any],
    scheme: str,
) -> CHestonRow:
    strike = product.get("strike", product.get("volatility_strike"))
    if strike is None:
        raise ValueError("Heston product row requires a strike.")
    return CHestonRow(
        float(model["spot"]),
        float(model["risk_free_rate"]),
        float(model.get("dividend_yield", 0.0)),
        float(model["initial_variance"]),
        float(model["kappa"]),
        float(model["theta"]),
        float(model["volatility_of_variance"]),
        float(model["rho"]),
        float(strike),
        float(product["maturity"]),
        int(row["seed"]),
        cpp_scheme_code(scheme),
    )


def raise_cpp_error(library: ctypes.CDLL, context: str) -> None:
    message = library.ai_factory_cuda_last_error()
    decoded = message.decode("utf-8") if message else "Unknown C++ error."
    raise RuntimeError(f"{context}: {decoded}")
