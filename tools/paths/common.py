"""Shared helpers for model path reconstruction tools."""

from __future__ import annotations

from tools.common.native_library import load_cpp_library as _load_cpp_library
from tools.common.time_grid import step_count_for_maturity


def load_cpp_library():
    return _load_cpp_library()


__all__ = ["load_cpp_library", "step_count_for_maturity"]
